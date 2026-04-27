# Coding Harness Demo

This repository is a small teaching lab for building a simple coding agent.
The agent is expected to work on the deliberately broken project in
`broken-repo/`, run its tests, inspect failures, edit code, and verify that the
tests pass.

## Repository Layout

- `Lab_CodingAgent.ipynb` - the notebook for the lab.
- `utils.py` - helper code for the lab (e.g., querying a local LLM).
- `broken-repo/` - the target repository the agent should repair.
- `broken-repo/tests/` - the failing tests that define the exercise goal.

## Local LLM Setup

The lab currently uses Ollama only. Install and start Ollama, then pull the
small local coding model:

```bash
ollama pull qwen2.5-coder:1.5b
```

Install the notebook's Python client dependency:

```bash
python3 -m pip install -U openai
```

The helper in `utils.py` connects to Ollama's OpenAI-compatible local endpoint:

```python
from utils import load_llm

llm = load_llm()
print(llm("Explain what a failing unit test means in one sentence."))
```

## Exercise Workflow

Start from the repository root, then inspect and test the broken project:

```bash
cd broken-repo
python3 -m unittest discover -s tests
```

The expected initial state is a failing test suite. A coding agent should use
the test output and source files under `broken-repo/tasklet/` to identify and
fix the bug, then rerun the same test command until the suite passes.

## Notes For Coding Agents

- Treat `broken-repo/` as the project under repair.
- Keep lab infrastructure files such as `Lab_CodingAgent.ipynb` and `utils.py`
  separate from the exercise unless the user asks to change the lab itself.
- Prefer small, verifiable edits and always rerun the relevant tests after
  changing code.
- The root project is intentionally lightweight; there is no hosted API key or
  remote LLM dependency required for the current version.
