# Arize AX Tracing — Integration Proposal

## Context

Add Arize AX LLM tracing to `autoresearch-unified` so every Claude API call made by the
autonomous experiment loop is captured in the Arize dashboard (spans, token counts,
prompt/response content, latency).

The project uses `anthropic>=0.40.0` with a synchronous `anthropic.Anthropic` client
inside `ClaudeBackend` (`tui/llm_backend.py`). No LLM tracing exists in this project
today. Per Arize docs, tracing must be initialized **before** the Anthropic client is
created.

---

## New Packages

```bash
pip install arize-otel openinference-instrumentation-anthropic
```

---

## Files Changed

| File | Action |
|------|--------|
| `pyproject.toml` | Add `tracing` optional extras group |
| `tui/tracing.py` | **Create** — tracing setup module |
| `tui/llm_backend.py` | Call `setup_arize_tracing()` before Anthropic client creation |
| `.env.example` | **Create** — document `ARIZE_SPACE_ID` / `ARIZE_API_KEY` |

---

## Implementation

### `pyproject.toml` — new optional extras

```toml
[project.optional-dependencies]
tracing = [
    "arize-otel",
    "openinference-instrumentation-anthropic",
]
```

Install together with the agent extras:

```bash
pip install -e ".[agent,tracing]"
```

---

### `tui/tracing.py` (new file)

```python
import os

_initialized = False


def setup_arize_tracing(project_name: str = "autoresearch-unified") -> bool:
    """Initialize Arize AX tracing. No-op if env vars are absent or packages missing.

    Requires environment variables:
        ARIZE_SPACE_ID  — from https://app.arize.com/organizations/-/settings/space-api-keys
        ARIZE_API_KEY   — from the same page

    Returns:
        True if tracing was successfully configured, False otherwise.
    """
    global _initialized
    if _initialized:
        return True

    space_id = os.environ.get("ARIZE_SPACE_ID")
    api_key = os.environ.get("ARIZE_API_KEY")
    if not space_id or not api_key:
        return False

    try:
        from arize.otel import register
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        tracer_provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
        )
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        _initialized = True
        return True
    except ImportError:
        return False
```

---

### `tui/llm_backend.py` — `ClaudeBackend.__init__()` change

Insert immediately **before** `self._client = anthropic.Anthropic(api_key=cred.api_key)`:

```python
from tui.tracing import setup_arize_tracing
setup_arize_tracing()
```

---

### `.env.example` (new file)

```bash
# Arize AX Tracing (optional — omit to disable)
ARIZE_SPACE_ID=your_space_id_here
ARIZE_API_KEY=your_api_key_here
```

---

## Design Notes

- **Graceful degradation** — If env vars are absent or packages not installed,
  `setup_arize_tracing()` returns `False` silently. No crash, no behavior change.
- **Idempotent** — Module-level `_initialized` flag prevents double-registration if
  `ClaudeBackend` is ever re-instantiated.
- **No business logic changes** — Tracing is purely additive. No changes to prompt
  construction, response parsing, retry logic, or experiment flow.
- **Thread-safe** — `AnthropicInstrumentor` wraps the SDK globally; synchronous calls
  made in the orchestrator's background thread are captured automatically.

---

## Verification Steps

1. Set credentials:
   ```bash
   export ARIZE_SPACE_ID=<your-space-id>
   export ARIZE_API_KEY=<your-api-key>
   ```
2. Install deps: `pip install -e ".[agent,tracing]"`
3. Run: `python -m tui.app --mode agent`
4. Open [https://app.arize.com](https://app.arize.com) → navigate to the
   `autoresearch-unified` project → confirm traces appear with model name, token counts,
   and prompt/response content.
