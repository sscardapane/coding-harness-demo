"""Ollama-backed LLM helper for the coding-agent lab.

The notebook keeps the interface deliberately small:

    from utils import load_llm

    llm = load_llm()
    print(llm("Explain this failing test in one sentence."))

Before running it, make sure Ollama is installed, running, and has the default
model available:

    ollama pull qwen2.5-coder:1.5b
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

DEFAULT_MODEL = "qwen2.5-coder:1.5b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_SYSTEM_PROMPT = "You are a concise coding assistant."

INSTALL_HINT = """Install and start Ollama, then pull the lab model:
ollama pull qwen2.5-coder:1.5b

The notebook also needs the Python OpenAI client:
pip install -U openai
"""


def _ollama_api_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def _ollama_tags_url(base_url: str) -> str:
    return f"{_ollama_api_base(base_url)}/api/tags"


def _installed_ollama_models(base_url: str) -> set[str]:
    try:
        with urllib.request.urlopen(_ollama_tags_url(base_url), timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            "Could not reach Ollama at "
            f"{_ollama_api_base(base_url)}. Start Ollama and try again.\n\n"
            f"{INSTALL_HINT}"
        ) from exc

    return {
        model["name"]
        for model in payload.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    }


def check_ollama_model(model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL) -> None:
    """Raise a helpful error if Ollama is unavailable or the model is missing."""

    installed = _installed_ollama_models(base_url)
    if model in installed:
        return

    available = ", ".join(sorted(installed)) or "none"
    raise RuntimeError(
        f"Ollama is running, but `{model}` is not installed.\n"
        f"Run: ollama pull {model}\n"
        f"Available models: {available}"
    )


@dataclass
class LabLLM:
    """Tiny chat-completions wrapper used throughout the notebook."""

    client: Any
    model: str
    temperature: float = 0.2
    max_tokens: int = 512
    top_p: float | None = None

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
        **kwargs: Any,
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "temperature": self.temperature if temperature is None else temperature,
        }

        resolved_top_p = self.top_p if top_p is None else top_p
        if resolved_top_p is not None:
            request["top_p"] = resolved_top_p

        request.update(kwargs)
        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content
        return content or ""


def load_llm(
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = "ollama",
    check_model: bool = True,
    **generation_defaults: Any,
) -> LabLLM:
    """Return a simple callable client for a local Ollama model."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI client first:\npip install -U openai") from exc

    if check_model:
        check_ollama_model(model=model, base_url=base_url)

    client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key)
    return LabLLM(client=client, model=model, **generation_defaults)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "LabLLM",
    "check_ollama_model",
    "load_llm",
]
