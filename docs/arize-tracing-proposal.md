# Arize AX Tracing — Integration Proposal

## Context

Add Arize AX LLM tracing to `autoresearch-unified` so every LLM API call made by the
autonomous experiment loop is captured in the Arize dashboard (spans, token counts,
prompt/response content, latency).

The project has five LLM backends (`tui/llm_backend.py`), four of which are active:

| Backend | SDK | Instrumentor |
|---------|-----|-------------|
| `ClaudeBackend` | `anthropic` | `openinference-instrumentation-anthropic` |
| `OpenRouterBackend` | `openai` (custom `base_url`) | `openinference-instrumentation-openai` |
| `OpenAIBackend` | `openai` | `openinference-instrumentation-openai` |
| `AzureOpenAIBackend` | `openai` (`AzureOpenAI` subclass) | `openinference-instrumentation-openai` |

No LLM tracing exists in this project today. Per Arize docs, tracing must be initialized
**before** any SDK client is created. The `OpenAIInstrumentor` monkey-patches the `openai`
module globally, so a single registration covers OpenRouter, OpenAI, and Azure backends
regardless of `base_url`.

---

## New Packages

```bash
pip install arize-otel openinference-instrumentation-openai openinference-instrumentation-anthropic
```

---

## Files Changed

| File | Action |
|------|--------|
| `pyproject.toml` | Add `tracing` optional extras group |
| `tui/tracing.py` | **Create** — tracing setup module (dual instrumentor) |
| `tui/llm_backend.py` | Call `setup_arize_tracing()` in `get_llm_backend()` factory before any backend constructor |
| `.env.example` | **Create** — document all env vars including `ARIZE_SPACE_ID` / `ARIZE_API_KEY` |

---

## Implementation

### `pyproject.toml` — new optional extras

```toml
[project.optional-dependencies]
tracing = [
    "arize-otel",
    "openinference-instrumentation-openai",
    "openinference-instrumentation-anthropic",
]
```

Install together with your backend extras:

```bash
# OpenRouter users
pip install -e ".[agent-openrouter,tracing]"

# Claude users
pip install -e ".[agent,tracing]"

# All backends + tracing
pip install -e ".[agent-all,tracing]"
```

---

### `tui/tracing.py` (new file)

```python
import os
import logging

logger = logging.getLogger(__name__)

_initialized = False


def setup_arize_tracing(project_name: str = "autoresearch-unified") -> bool:
    """Initialize Arize AX tracing for OpenAI and Anthropic SDKs.

    No-op if env vars are absent or packages not installed.

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
    except ImportError:
        logger.debug("arize-otel not installed, skipping tracing")
        return False

    tracer_provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
    )

    # Instrument OpenAI SDK (covers OpenRouterBackend, OpenAIBackend, AzureOpenAIBackend)
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Arize tracing: OpenAI SDK instrumented")
    except ImportError:
        logger.debug("openinference-instrumentation-openai not installed, skipping OpenAI tracing")

    # Instrument Anthropic SDK (covers ClaudeBackend)
    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info("Arize tracing: Anthropic SDK instrumented")
    except ImportError:
        logger.debug("openinference-instrumentation-anthropic not installed, skipping Anthropic tracing")

    _initialized = True
    return True
```

---

### `tui/llm_backend.py` — `get_llm_backend()` change

Insert at the **top** of `get_llm_backend()`, before any backend constructor runs:

```python
# Initialize Arize AX tracing (no-op if env vars or packages missing).
# Must run BEFORE any OpenAI/Anthropic client is created.
from tui.tracing import setup_arize_tracing
setup_arize_tracing()
```

This replaces the earlier approach of calling tracing inside `ClaudeBackend.__init__()`.
Placing it in the factory ensures every backend — including OpenRouter — is instrumented
before its SDK client is instantiated.

---

### `.env.example` (new file)

```bash
# ──────────────────────────────────────────────
# LLM Backend credentials (set ONE group)
# ──────────────────────────────────────────────

# Anthropic Claude (recommended)
# ANTHROPIC_API_KEY=sk-ant-...

# OpenRouter (200+ models, one key)
# OPENROUTER_API_KEY=sk-or-...
# OPENROUTER_MODEL=anthropic/claude-sonnet-4.6

# OpenAI
# OPENAI_API_KEY=sk-...

# Azure OpenAI
# AZURE_OPENAI_API_KEY=...
# AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# ──────────────────────────────────────────────
# Arize AX Tracing (optional — omit to disable)
# ──────────────────────────────────────────────
# ARIZE_SPACE_ID=your_space_id_here
# ARIZE_API_KEY=your_api_key_here
```

---

## Design Notes

- **Graceful degradation** — If env vars are absent or packages not installed,
  `setup_arize_tracing()` returns `False` silently. No crash, no behavior change.
- **Partial instrumentation** — Each instrumentor has its own try/except. If only
  `openinference-instrumentation-openai` is installed, OpenRouter/OpenAI/Azure calls
  are traced while Claude calls are not (and vice versa).
- **Idempotent** — Module-level `_initialized` flag prevents double-registration if
  `get_llm_backend()` is called more than once.
- **No business logic changes** — Tracing is purely additive. No changes to prompt
  construction, response parsing, retry logic, or experiment flow.
- **Thread-safe** — Both instrumentors wrap their SDKs globally; synchronous calls
  made in the orchestrator's background thread are captured automatically.
- **OpenRouter compatibility** — `OpenAIInstrumentor` patches the `openai` module at
  the SDK level, intercepting `chat.completions.create()` regardless of `base_url`.
  OpenRouter calls appear in Arize with the full model name (e.g. `anthropic/claude-sonnet-4.6`).

---

## Verification Steps

1. Set credentials:
   ```bash
   export ARIZE_SPACE_ID=<your-space-id>
   export ARIZE_API_KEY=<your-api-key>
   export OPENROUTER_API_KEY=sk-or-...
   ```
2. Install deps: `pip install -e ".[agent-openrouter,tracing]"`
3. Run: `python -m tui.app --mode agent`
4. Open [https://app.arize.com](https://app.arize.com) → navigate to the
   `autoresearch-unified` project → confirm traces appear with model name, token counts,
   and prompt/response content.
5. Verify graceful degradation: unset `ARIZE_SPACE_ID` and `ARIZE_API_KEY`, confirm
   the app runs normally with no errors.
