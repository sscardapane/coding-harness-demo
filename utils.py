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
import html as html_lib
import re
import subprocess
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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


@dataclass
class AgentTraceStep:
    """One tool call made by the notebook agent."""

    step: int
    tool: str
    args: Mapping[str, Any]
    note: str
    observation: str
    model_reply: str = ""
    status: str = "ok"

    @property
    def observation_chars(self) -> int:
        return len(self.observation)


class AgentTrace:
    """Collect agent steps and render a compact notebook trace."""

    def __init__(self) -> None:
        self.steps: list[AgentTraceStep] = []

    def add(
        self,
        step: int,
        action: Mapping[str, Any],
        observation: str,
        model_reply: str = "",
    ) -> AgentTraceStep:
        tool = str(action.get("tool") or "unknown")
        args = action.get("args") or {}
        if not isinstance(args, Mapping):
            args = {"value": args}
        note = str(action.get("note") or action.get("thought") or "")

        trace_step = AgentTraceStep(
            step=step,
            tool=tool,
            args=args,
            note=note,
            observation=str(observation),
            model_reply=model_reply,
            status=_infer_step_status(tool, str(observation)),
        )
        self.steps.append(trace_step)
        return trace_step

    @property
    def tool_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.tool] = counts.get(step.tool, 0) + 1
        return counts

    @property
    def last_status(self) -> str:
        if not self.steps:
            return "not started"
        return self.steps[-1].status

    def to_text(self) -> str:
        rows = []
        for step in self.steps:
            summary = summarize_observation(step.tool, step.observation)
            rows.append(f"{step.step}. {step.tool}: {summary}")
        return "\n".join(rows) or "(no agent steps yet)"


def summarize_observation(tool: str, observation: str, max_chars: int = 260) -> str:
    """Return a short human-readable observation summary for trace tables."""

    text = observation.strip()
    if not text:
        return "(empty)"

    if tool == "run_tests":
        return _summarize_test_output(text, max_chars=max_chars)

    if tool == "apply_patch":
        if text.startswith("Patch applied"):
            return "Patch applied."
        if text.startswith("Patch failed"):
            stderr = _extract_section(text, "STDERR")
            return _truncate("Patch failed. " + (stderr or text), max_chars)

    if tool == "read_file":
        visible_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("...")
        ]
        return _truncate(" | ".join(visible_lines[:3]), max_chars)

    return _truncate(" ".join(text.split()), max_chars)


def extract_patch_text(text: str) -> str:
    """Accept raw patches or patches wrapped in Markdown fences."""

    fenced = re.search(r"```(?:diff|patch)?\s*\n(.*?)```", text, flags=re.DOTALL)
    patch = fenced.group(1) if fenced else text
    return patch.strip() + "\n"


def repair_unified_diff_hunk_counts(patch: str) -> str:
    """Fix simple unified-diff hunk counts to match the hunk body."""

    lines = patch.splitlines()
    repaired: list[str] = []
    index = 0
    header_pattern = re.compile(
        r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
        r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<suffix>.*)$"
    )

    while index < len(lines):
        line = lines[index]
        match = header_pattern.match(line)
        if not match:
            repaired.append(line)
            index += 1
            continue

        body: list[str] = []
        old_count = 0
        new_count = 0
        index += 1

        while index < len(lines):
            candidate = lines[index]
            next_is_file_header = (
                candidate.startswith("--- ")
                and index + 1 < len(lines)
                and lines[index + 1].startswith("+++ ")
            )
            if candidate.startswith("@@ ") or candidate.startswith("diff --git ") or next_is_file_header:
                break

            body.append(candidate)
            if candidate.startswith("\\"):
                index += 1
                continue
            if candidate.startswith(" "):
                old_count += 1
                new_count += 1
            elif candidate.startswith("-"):
                old_count += 1
            elif candidate.startswith("+"):
                new_count += 1
            index += 1

        repaired.append(
            f"@@ -{match.group('old_start')},{old_count} "
            f"+{match.group('new_start')},{new_count} @@{match.group('suffix')}"
        )
        repaired.extend(body)

    return "\n".join(repaired).strip() + "\n"


def normalize_patch_paths(repo: str | Path, patch: str) -> str:
    """Rewrite patch file headers to existing repo-relative paths when obvious."""

    repo_path = Path(repo)
    normalized = []
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            normalized.append(_normalize_diff_git_line(repo_path, line))
        elif line.startswith("--- "):
            normalized.append(_normalize_file_header_line(repo_path, line, marker="---", side="a"))
        elif line.startswith("+++ "):
            normalized.append(_normalize_file_header_line(repo_path, line, marker="+++", side="b"))
        else:
            normalized.append(line)
    return "\n".join(normalized).strip() + "\n"


def apply_unified_patch(repo: str | Path, patch: str) -> str:
    """Apply a unified diff, retrying after small LLM-oriented repairs."""

    repo_path = Path(repo)
    original_patch = extract_patch_text(patch)
    candidates = _patch_candidates(repo_path, original_patch)
    attempts: list[tuple[str, int, subprocess.CompletedProcess[str]]] = []
    last_patch = original_patch

    for label, candidate in candidates:
        for strip_level in (0, 1, 2):
            dry_run = _run_patch_command(repo_path, candidate, strip_level, dry_run=True)
            attempts.append((label, strip_level, dry_run))
            last_patch = candidate

            if dry_run.returncode != 0:
                continue

            result = _run_patch_command(repo_path, candidate, strip_level, dry_run=False)
            attempts.append((label, strip_level, result))
            if result.returncode != 0:
                continue

            if label == "original":
                return f"Patch applied with patch -p{strip_level}. Run the tests next."
            return f"Patch applied after {label} with patch -p{strip_level}. Run the tests next."

    last_label, last_strip, last_result = attempts[-1]
    return _format_patch_failure(
        last_result,
        repaired_patch=last_patch if last_patch != original_patch else None,
        first_result=attempts[0][2] if attempts[0][2] is not last_result else None,
        attempted_labels=[f"{label} using patch -p{strip}" for label, strip, _ in attempts],
    )


def render_agent_trace(trace: AgentTrace) -> str:
    """Render an AgentTrace as dependency-free HTML for notebooks."""

    steps = trace.steps
    total_chars = sum(step.observation_chars for step in steps)
    patch_count = trace.tool_counts.get("apply_patch", 0)
    status_label = _status_label(trace.last_status)

    return f"""
    <style>
    .agent-trace {{
      --border: #d8dee9;
      --muted: #667085;
      --bg-soft: #f8fafc;
      --text: #111827;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
    }}
    .agent-summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(110px, 1fr));
      gap: 8px;
      margin: 8px 0 12px;
    }}
    .agent-card {{
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      background: white;
    }}
    .agent-card strong {{
      display: block;
      font-size: 18px;
      line-height: 1.2;
    }}
    .agent-card span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .agent-timeline {{
      display: flex;
      gap: 6px;
      overflow-x: auto;
      padding: 4px 0 10px;
    }}
    .agent-chip {{
      min-width: 58px;
      border-radius: 8px;
      color: white;
      padding: 6px 7px;
      line-height: 1.15;
      box-sizing: border-box;
    }}
    .agent-chip b {{
      display: block;
      font-size: 13px;
    }}
    .agent-chip span {{
      display: block;
      font-size: 10px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .agent-plots {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) minmax(260px, 2fr);
      gap: 12px;
      margin: 0 0 12px;
    }}
    .agent-plot {{
      border: 1px solid var(--border);
      border-radius: 8px;
      background: white;
      padding: 10px;
    }}
    .agent-plot-title {{
      font-weight: 600;
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .agent-bar-row {{
      display: grid;
      grid-template-columns: 90px 1fr 42px;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      margin: 5px 0;
    }}
    .agent-bar-track {{
      height: 9px;
      background: #eef2f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .agent-bar-fill {{
      height: 100%;
      min-width: 2px;
      border-radius: 999px;
    }}
    .agent-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    .agent-table th,
    .agent-table td {{
      border-top: 1px solid var(--border);
      padding: 8px;
      vertical-align: top;
      text-align: left;
    }}
    .agent-table th {{
      color: var(--muted);
      font-weight: 600;
      background: var(--bg-soft);
    }}
    .agent-pill {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 7px;
      color: white;
      font-size: 11px;
      font-weight: 600;
      white-space: nowrap;
    }}
    .agent-muted {{
      color: var(--muted);
    }}
    .agent-pre {{
      max-height: 260px;
      overflow: auto;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 10px;
      white-space: pre-wrap;
      font-size: 12px;
    }}
    .agent-details summary {{
      cursor: pointer;
      color: #2563eb;
      margin-top: 6px;
    }}
    .agent-render-note {{
      color: var(--muted);
      font-size: 12px;
      margin: 0 0 8px;
    }}
    .agent-short {{
      color: var(--muted);
      font-size: 11px;
      margin-top: 3px;
    }}
    @media (max-width: 860px) {{
      .agent-summary,
      .agent-plots {{
        grid-template-columns: 1fr;
      }}
    }}
    </style>
    <div class="agent-trace">
      <div class="agent-summary">
        {_summary_card(len(steps), "steps")}
        {_summary_card(status_label, "latest status")}
        {_summary_card(patch_count, "patch attempts")}
        {_summary_card(total_chars, "output chars")}
      </div>
      {_render_timeline(steps)}
      <div class="agent-plots">
        {_render_tool_counts(trace.tool_counts)}
        {_render_output_lengths(steps)}
      </div>
      <div class="agent-render-note">
        Table cells show shortened summaries. Open details for the full action args,
        model reply, and observation.
      </div>
      {_render_trace_table(steps)}
    </div>
    """


def display_agent_trace(trace: AgentTrace, *, clear: bool = False) -> str:
    """Display an agent trace in a notebook, or return HTML outside IPython."""

    html = render_agent_trace(trace)
    try:
        from IPython.display import HTML, clear_output, display
    except ImportError:
        print(trace.to_text())
        return html

    if clear:
        clear_output(wait=True)
    display(HTML(html))
    return html


def format_raw_agent_step(
    step: int,
    model_reply: str,
    tool: str,
    observation: str,
    *,
    max_observation_chars: int | None = 5000,
) -> str:
    """Format one agent step in the original plain-text lab style."""

    output = observation
    truncated = False
    if max_observation_chars is not None and len(output) > max_observation_chars:
        output = output[:max_observation_chars]
        truncated = True

    suffix = "\n\n... [observation truncated in raw output]" if truncated else ""
    return (
        f"\n===== Step {step}: model =====\n"
        f"{model_reply}\n"
        f"\n===== Step {step}: observation from {tool} =====\n"
        f"{output}{suffix}"
    )


def print_raw_agent_step(
    step: int,
    model_reply: str,
    tool: str,
    observation: str,
    *,
    max_observation_chars: int | None = 5000,
) -> None:
    """Print one raw agent step for didactic notebook runs."""

    print(
        format_raw_agent_step(
            step,
            model_reply,
            tool,
            observation,
            max_observation_chars=max_observation_chars,
        )
    )


def _patch_candidates(repo: Path, patch: str) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, candidate: str) -> None:
        candidate = candidate.strip() + "\n"
        if candidate not in seen:
            candidates.append((label, candidate))
            seen.add(candidate)

    normalized = normalize_patch_paths(repo, patch)
    repaired = repair_unified_diff_hunk_counts(patch)
    normalized_repaired = repair_unified_diff_hunk_counts(normalized)

    add("original", patch)
    add("normalizing patch paths", normalized)
    add("repairing unified-diff hunk counts", repaired)
    add("normalizing patch paths and repairing unified-diff hunk counts", normalized_repaired)

    return candidates


def _normalize_diff_git_line(repo: Path, line: str) -> str:
    parts = line.split()
    if len(parts) < 4:
        return line

    old_path = _resolve_patch_path(repo, parts[2])
    new_path = _resolve_patch_path(repo, parts[3])
    if old_path is None and new_path is None:
        return line

    parts[2] = f"a/{old_path or new_path}"
    parts[3] = f"b/{new_path or old_path}"
    return " ".join(parts)


def _normalize_file_header_line(repo: Path, line: str, *, marker: str, side: str) -> str:
    path, rest = _split_patch_header_path(line[4:])
    resolved = _resolve_patch_path(repo, path)
    if resolved is None:
        return line
    return f"{marker} {side}/{resolved}{rest}"


def _split_patch_header_path(text: str) -> tuple[str, str]:
    if not text:
        return text, ""
    if text.startswith('"'):
        end = text.find('"', 1)
        if end != -1:
            return text[1:end], text[end + 1 :]
    parts = text.split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " " + parts[1]


def _resolve_patch_path(repo: Path, patch_path: str) -> str | None:
    if patch_path == "/dev/null":
        return None

    clean = patch_path.strip().strip('"')
    if clean.startswith("a/") or clean.startswith("b/"):
        clean = clean[2:]
    clean = clean.lstrip("/")

    candidates = [clean]
    repo_name = repo.name
    if clean.startswith(f"{repo_name}/"):
        candidates.append(clean[len(repo_name) + 1 :])

    for candidate in candidates:
        if candidate and (repo / candidate).is_file():
            return candidate

    basename = Path(clean).name
    if not basename:
        return None

    matches = [
        path.relative_to(repo).as_posix()
        for path in repo.rglob(basename)
        if path.is_file()
        and ".git" not in path.relative_to(repo).parts
        and "__pycache__" not in path.relative_to(repo).parts
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _run_patch_command(
    repo: Path,
    patch: str,
    strip_level: int,
    *,
    dry_run: bool,
) -> subprocess.CompletedProcess[str]:
    command = ["patch", f"-p{strip_level}", "--batch", "--forward"]
    if dry_run:
        command.append("--dry-run")
    return subprocess.run(
        command,
        cwd=repo,
        input=patch,
        check=False,
        capture_output=True,
        text=True,
    )


def _format_patch_failure(
    result: subprocess.CompletedProcess[str],
    *,
    repaired_patch: str | None = None,
    first_result: subprocess.CompletedProcess[str] | None = None,
    attempted_labels: Sequence[str] | None = None,
) -> str:
    details = textwrap.dedent(
        f"""
        Patch failed with exit code {result.returncode}.

        STDOUT:
        {result.stdout.rstrip() or "(empty)"}

        STDERR:
        {result.stderr.rstrip() or "(empty)"}
        """
    ).strip()

    if first_result is not None:
        details += textwrap.dedent(
            f"""

            Initial patch dry-run error:
            {first_result.stderr.rstrip() or "(empty)"}
            """
        ).rstrip()

    if attempted_labels:
        details += "\n\nPatch variants tried: " + ", ".join(attempted_labels)

    if repaired_patch is not None:
        details += textwrap.dedent(
            f"""

            Last repaired patch attempted:
            {repaired_patch.rstrip()}
            """
        ).rstrip()

    return details


def _infer_step_status(tool: str, observation: str) -> str:
    text = observation.strip()
    if tool == "finish":
        return "pass"
    if text.startswith("ERROR:") or text.startswith("Patch failed"):
        return "error"
    if tool == "run_tests":
        if re.search(r"exit code:\s*0\b", text) or re.search(r"(?m)^OK$", text):
            return "pass"
        if "FAILED" in text or re.search(r"exit code:\s*[1-9]", text):
            return "fail"
    return "ok"


def _status_label(status: str) -> str:
    labels = {
        "pass": "passing",
        "fail": "failing",
        "error": "error",
        "ok": "running",
    }
    return labels.get(status, status)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _extract_section(text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}:\n(.*?)(?:\n\n[A-Z]+:|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def _summarize_test_output(text: str, max_chars: int) -> str:
    interesting: list[str] = []
    exit_match = re.search(r"exit code:\s*(-?\d+)", text)
    if exit_match:
        interesting.append(f"exit {exit_match.group(1)}")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (
            line.startswith(("FAIL:", "ERROR:", "AssertionError:"))
            or line.startswith(("FAILED", "Ran "))
            or line == "OK"
        ):
            interesting.append(line)

    if not interesting:
        interesting.append(" ".join(text.split()))
    return _truncate(" | ".join(interesting), max_chars)


def _escape(value: Any) -> str:
    return html_lib.escape(str(value), quote=True)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _tool_color(tool: str) -> str:
    colors = {
        "inspect_files": "#2563eb",
        "read_file": "#0f766e",
        "apply_patch": "#7c3aed",
        "run_tests": "#ea580c",
        "finish": "#16a34a",
        "parser_error": "#dc2626",
    }
    return colors.get(tool, "#475569")


def _status_color(status: str) -> str:
    colors = {
        "pass": "#16a34a",
        "fail": "#dc2626",
        "error": "#b91c1c",
        "ok": "#475569",
    }
    return colors.get(status, "#475569")


def _summary_card(value: Any, label: str) -> str:
    return (
        '<div class="agent-card">'
        f"<strong>{_escape(value)}</strong>"
        f"<span>{_escape(label)}</span>"
        "</div>"
    )


def _render_timeline(steps: Sequence[AgentTraceStep]) -> str:
    if not steps:
        return '<div class="agent-muted">No agent steps yet.</div>'

    chips = []
    for step in steps:
        chips.append(
            '<div class="agent-chip" '
            f'style="background:{_tool_color(step.tool)}" '
            f'title="{_escape(step.tool)}">'
            f"<b>{step.step}</b>"
            f"<span>{_escape(step.tool)}</span>"
            "</div>"
        )
    return f'<div class="agent-timeline">{"".join(chips)}</div>'


def _render_tool_counts(counts: Mapping[str, int]) -> str:
    if not counts:
        return '<div class="agent-plot"><div class="agent-plot-title">Tool Calls</div>No data.</div>'

    max_count = max(counts.values()) or 1
    rows = []
    for tool, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        width = max(3, int(count / max_count * 100))
        rows.append(
            '<div class="agent-bar-row">'
            f"<div>{_escape(tool)}</div>"
            '<div class="agent-bar-track">'
            f'<div class="agent-bar-fill" style="width:{width}%; background:{_tool_color(tool)}"></div>'
            "</div>"
            f"<div>{count}</div>"
            "</div>"
        )
    return (
        '<div class="agent-plot">'
        '<div class="agent-plot-title">Tool Calls</div>'
        f'{"".join(rows)}'
        "</div>"
    )


def _render_output_lengths(steps: Sequence[AgentTraceStep]) -> str:
    if not steps:
        return '<div class="agent-plot"><div class="agent-plot-title">Observation Size</div>No data.</div>'

    max_chars = max(step.observation_chars for step in steps) or 1
    rows = []
    for step in steps:
        width = max(3, int(step.observation_chars / max_chars * 100))
        rows.append(
            '<div class="agent-bar-row">'
            f"<div>step {step.step}</div>"
            '<div class="agent-bar-track">'
            f'<div class="agent-bar-fill" style="width:{width}%; background:{_tool_color(step.tool)}"></div>'
            "</div>"
            f"<div>{step.observation_chars}</div>"
            "</div>"
        )
    return (
        '<div class="agent-plot">'
        '<div class="agent-plot-title">Observation Size</div>'
        f'{"".join(rows)}'
        "</div>"
    )


def _render_trace_table(steps: Sequence[AgentTraceStep]) -> str:
    if not steps:
        return '<div class="agent-muted">No trace rows to show.</div>'

    rows = []
    for step in steps:
        summary = summarize_observation(step.tool, step.observation)
        args_text = _json_text(step.args)
        args_summary = _truncate(args_text, 160)
        args_hint = '<div class="agent-short">shortened; open details</div>' if args_text != args_summary else ""
        note_summary = _truncate(step.note, 180)
        note_hint = '<div class="agent-short">shortened; open details</div>' if step.note.strip() != note_summary else ""
        rows.append(
            "<tr>"
            f"<td>{step.step}</td>"
            "<td>"
            f'<span class="agent-pill" style="background:{_tool_color(step.tool)}">{_escape(step.tool)}</span>'
            "</td>"
            f"<td><code>{_escape(args_summary)}</code>{args_hint}</td>"
            f"<td>{_escape(note_summary)}{note_hint}</td>"
            "<td>"
            f"{_escape(summary)}"
            '<details class="agent-details">'
            "<summary>details</summary>"
            f'<div class="agent-muted">full action args</div><pre class="agent-pre">{_escape(args_text)}</pre>'
            f'<div class="agent-muted">model reply</div><pre class="agent-pre">{_escape(step.model_reply)}</pre>'
            f'<div class="agent-muted">observation</div><pre class="agent-pre">{_escape(step.observation)}</pre>'
            "</details>"
            "</td>"
            f"<td>{step.observation_chars}</td>"
            "<td>"
            f'<span class="agent-pill" style="background:{_status_color(step.status)}">{_escape(_status_label(step.status))}</span>'
            "</td>"
            "</tr>"
        )

    return (
        '<table class="agent-table">'
        "<thead><tr>"
        "<th>Step</th><th>Tool</th><th>Args</th><th>Note</th>"
        "<th>Observation Summary</th><th>Chars</th><th>Status</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table>"
    )


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
    "AgentTrace",
    "AgentTraceStep",
    "LabLLM",
    "apply_unified_patch",
    "check_ollama_model",
    "display_agent_trace",
    "extract_patch_text",
    "format_raw_agent_step",
    "load_llm",
    "normalize_patch_paths",
    "print_raw_agent_step",
    "repair_unified_diff_hunk_counts",
    "render_agent_trace",
    "summarize_observation",
]
