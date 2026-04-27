"""Small local LLM helper for the coding-agent lab.

The notebook uses this module to hide the server/client details from students:

    from utils import load_llm

    llm = load_llm()
    print(llm("Explain this failing test in one sentence."))

By default this starts Qwen/Qwen3.5-2B with the lightweight Hugging Face
Transformers server and talks to it through the OpenAI-compatible API.
"""

from __future__ import annotations

import atexit
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_SYSTEM_PROMPT = "You are a concise coding assistant."
DEFAULT_LOG_FILE = Path(tempfile.gettempdir()) / "coding_agent_llm_server.log"

INSTALL_HINT = """Install the lab LLM dependencies first:
!pip -q install -U openai pillow torchvision
!pip -q install -U "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
"""

_SERVER_PROCESS: subprocess.Popen[bytes] | None = None


def _models_endpoint(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/models"


def _server_is_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(_models_endpoint(base_url), timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def start_llm_server(
    model: str = DEFAULT_MODEL,
    *,
    port: int = 8000,
    timeout_s: int = 600,
    log_file: str | Path = DEFAULT_LOG_FILE,
    extra_args: Sequence[str] | None = None,
) -> str:
    """Start a local Transformers server and return its OpenAI base URL."""

    global _SERVER_PROCESS

    base_url = f"http://127.0.0.1:{port}/v1"
    if _server_is_ready(base_url):
        return base_url

    if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
        return base_url

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "transformers",
        "serve",
        "--force-model",
        model,
        "--port",
        str(port),
        "--continuous-batching",
    ]
    if extra_args:
        command.extend(extra_args)

    try:
        with log_path.open("ab") as log_handle:
            _SERVER_PROCESS = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Could not find the `transformers` command.\n\n{INSTALL_HINT}"
        ) from exc

    atexit.register(stop_llm_server)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _SERVER_PROCESS.poll() is not None:
            log_tail = _tail(log_path)
            raise RuntimeError(
                "The LLM server exited before becoming ready."
                f"\n\nLast lines from {log_path}:\n{log_tail}"
            )
        if _server_is_ready(base_url):
            return base_url
        time.sleep(5)

    stop_llm_server()
    log_tail = _tail(log_path)
    raise TimeoutError(
        f"The LLM server did not become ready within {timeout_s} seconds."
        f"\n\nLast lines from {log_path}:\n{log_tail}"
    )


def stop_llm_server() -> None:
    """Stop the server started by this module, if it is still running."""

    global _SERVER_PROCESS

    if _SERVER_PROCESS is None or _SERVER_PROCESS.poll() is not None:
        return

    _SERVER_PROCESS.terminate()
    try:
        _SERVER_PROCESS.wait(timeout=20)
    except subprocess.TimeoutExpired:
        _SERVER_PROCESS.kill()
        _SERVER_PROCESS.wait(timeout=20)
    finally:
        _SERVER_PROCESS = None


@dataclass
class LabLLM:
    """Tiny chat-completions wrapper used throughout the notebook."""

    client: Any
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float | None = None
    top_k: int | None = 20
    presence_penalty: float | None = None
    enable_thinking: bool = False

    def __call__(
        self,
        prompt: str,
        *,
        system: str = DEFAULT_SYSTEM_PROMPT,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }

        resolved_top_p = self.top_p if top_p is None else top_p
        resolved_presence_penalty = (
            self.presence_penalty if presence_penalty is None else presence_penalty
        )
        resolved_top_k = self.top_k if top_k is None else top_k
        resolved_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )

        if resolved_top_p is not None:
            request["top_p"] = resolved_top_p
        if resolved_presence_penalty is not None:
            request["presence_penalty"] = resolved_presence_penalty

        extra_body: dict[str, Any] = {}
        if resolved_top_k is not None:
            extra_body["top_k"] = resolved_top_k
        if resolved_enable_thinking and "Qwen3.5" in self.model:
            extra_body["enable_thinking"] = True
        if extra_body:
            request["extra_body"] = extra_body

        request.update(kwargs)
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        return content or ""


def load_llm(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str | None = None,
    api_key: str = "EMPTY",
    start_server: bool = True,
    port: int = 8000,
    timeout_s: int = 600,
    log_file: str | Path = DEFAULT_LOG_FILE,
    **generation_defaults: Any,
) -> LabLLM:
    """Load the lab LLM and return a simple callable client.

    Set ``start_server=False`` when using an already-running OpenAI-compatible
    endpoint, for example Ollama:

        llm = load_llm(
            model="qwen2.5-coder:1.5b",
            base_url="http://127.0.0.1:11434/v1",
            start_server=False,
        )
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(INSTALL_HINT) from exc

    if base_url is None:
        if start_server:
            base_url = start_llm_server(
                model=model,
                port=port,
                timeout_s=timeout_s,
                log_file=log_file,
            )
        else:
            base_url = DEFAULT_BASE_URL

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    return LabLLM(client=client, model=model, **generation_defaults)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_LOG_FILE",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "LabLLM",
    "load_llm",
    "start_llm_server",
    "stop_llm_server",
]
