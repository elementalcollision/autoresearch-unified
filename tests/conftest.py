"""Shared test fixtures."""
import sys
from pathlib import Path

import pytest

# Ensure the project root is on the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Env vars that could trigger real backend selection or API calls
_LLM_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "OLLAMA_MODEL",
    "CLAUDE_MODEL",
    "OPENAI_MODEL",
]


@pytest.fixture
def clean_llm_env(monkeypatch):
    """Remove all LLM-related env vars so tests never leak real credentials."""
    for var in _LLM_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def sample_hw_info():
    """Consistent fake hardware dict for system-prompt tests."""
    return {
        "chip_name": "Test GPU",
        "memory_gb": 16,
        "gpu_cores": 64,
        "peak_tflops": 10.0,
        "chip_tier": "test",
    }
