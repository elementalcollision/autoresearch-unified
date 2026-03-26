"""Tests for tui.llm_backend — response parsing, system prompt, and backend factory."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# The module-level SYSTEM_PROMPT = get_system_prompt() calls backends.get_hardware_info()
# at import time. Patch it so import succeeds without real GPU detection.
_FAKE_HW = {
    "chip_name": "Test GPU",
    "memory_gb": 16,
    "gpu_cores": 64,
    "peak_tflops": 10.0,
    "chip_tier": "test",
}
with patch("backends.get_hardware_info", return_value=_FAKE_HW):
    from tui.llm_backend import (  # noqa: E402
        ExperimentProposal,
        get_system_prompt,
        parse_llm_response,
        OllamaBackend,
        get_llm_backend,
    )

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

VALID_RESPONSE = """\
DESCRIPTION: Increase MATRIX_LR from 0.04 to 0.06
REASONING: Higher learning rate on the matrix branch may allow faster convergence \
within the 5-minute budget. Prior experiments show the current rate is conservative.
CODE:
# --- HP BLOCK START ---
MATRIX_LR = 0.06
BATCH_SIZE = 64
# --- HP BLOCK END ---"""

RESPONSE_WITH_FENCES = """\
DESCRIPTION: Lower batch size to 32
REASONING: Smaller batches give more gradient steps in the fixed time budget.
CODE:
```python
# --- HP BLOCK START ---
BATCH_SIZE = 32
# --- HP BLOCK END ---
```"""

MULTILINE_REASONING = """\
DESCRIPTION: Test change
REASONING: Line one of reasoning.
Line two of reasoning.
Line three of reasoning.
CODE:
x = 1"""


# ---------------------------------------------------------------------------
# parse_llm_response — pure logic, no mocking
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_valid_response(self):
        result = parse_llm_response(VALID_RESPONSE)
        assert isinstance(result, ExperimentProposal)
        assert result.description == "Increase MATRIX_LR from 0.04 to 0.06"
        assert "faster convergence" in result.reasoning
        assert "MATRIX_LR = 0.06" in result.code

    def test_markdown_fences_stripped(self):
        result = parse_llm_response(RESPONSE_WITH_FENCES)
        assert "```" not in result.code
        assert "BATCH_SIZE = 32" in result.code

    def test_missing_description_raises(self):
        text = "REASONING: reason\nCODE:\nx = 1"
        with pytest.raises(ValueError, match="DESCRIPTION"):
            parse_llm_response(text)

    def test_missing_reasoning_raises(self):
        text = "DESCRIPTION: desc\nCODE:\nx = 1"
        with pytest.raises(ValueError, match="REASONING"):
            parse_llm_response(text)

    def test_missing_code_raises(self):
        # REASONING regex requires CODE: as lookahead, so include CODE: but no body
        text = "DESCRIPTION: desc\nREASONING: reason\nCODE:"
        with pytest.raises(ValueError, match="CODE"):
            parse_llm_response(text)

    def test_multiline_reasoning(self):
        result = parse_llm_response(MULTILINE_REASONING)
        assert "Line one" in result.reasoning
        assert "Line three" in result.reasoning

    def test_code_with_comments(self):
        text = "DESCRIPTION: d\nREASONING: r\nCODE:\n# comment\nx = 1"
        result = parse_llm_response(text)
        assert "# comment" in result.code

    def test_whitespace_stripped(self):
        text = "DESCRIPTION:   padded description   \nREASONING: r\nCODE:\nx = 1"
        result = parse_llm_response(text)
        assert result.description == "padded description"


# ---------------------------------------------------------------------------
# get_system_prompt — pass hw_info dict, check substrings
# ---------------------------------------------------------------------------

class TestGetSystemPrompt:
    def test_nvidia_platform(self):
        hw = {"chip_name": "NVIDIA H100", "memory_gb": 80, "chip_tier": "h100"}
        prompt = get_system_prompt(hw)
        assert "NVIDIA GPU" in prompt
        assert "torch.compile" in prompt

    def test_rocm_platform_v7(self):
        hw = {"chip_name": "AMD MI300X", "memory_gb": 192, "rocm_version": (7, 1)}
        prompt = get_system_prompt(hw)
        assert "ROCm 7.1" in prompt
        assert "HIP graph capture" in prompt

    def test_rocm_platform_v6(self):
        hw = {"chip_name": "AMD MI250", "memory_gb": 128, "rocm_version": (6, 2)}
        prompt = get_system_prompt(hw)
        assert "ROCm 6.2" in prompt
        assert "AMD Triton" in prompt

    def test_gaudi_platform(self):
        hw = {"chip_name": "Gaudi 3", "memory_gb": 128, "chip_tier": "gaudi3"}
        prompt = get_system_prompt(hw)
        assert "Intel Gaudi 3 HPU" in prompt
        assert "hpu_backend" in prompt

    def test_apple_silicon_default(self):
        hw = {"chip_name": "M4 Max", "memory_gb": 64, "chip_tier": "m4_max"}
        prompt = get_system_prompt(hw)
        assert "Apple Silicon" in prompt
        assert "MLX" in prompt

    def test_high_memory_note(self):
        hw = {"chip_name": "H100", "memory_gb": 80, "chip_tier": "test"}
        prompt = get_system_prompt(hw)
        assert "generous" in prompt

    def test_low_memory_note(self):
        hw = {"chip_name": "M4", "memory_gb": 16, "chip_tier": "test"}
        prompt = get_system_prompt(hw)
        assert "limited" in prompt


# ---------------------------------------------------------------------------
# Backend names
# ---------------------------------------------------------------------------

class TestBackendNames:
    def test_ollama_name(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.3")
        backend = OllamaBackend()
        assert backend.name() == "Ollama (llama3.3)"

    @patch("tui.credentials.resolve_api_key")
    def test_claude_name(self, mock_resolve, monkeypatch):
        from tui.credentials import CredentialSource
        mock_resolve.return_value = CredentialSource(api_key="sk-ant-fake", source="env")
        mock_anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        # Re-import to pick up the mock
        from tui.llm_backend import ClaudeBackend
        backend = ClaudeBackend()
        name = backend.name()
        assert "Claude" in name
        assert "env" in name


# ---------------------------------------------------------------------------
# Factory: get_llm_backend
# ---------------------------------------------------------------------------

class TestGetLLMBackend:
    def test_ollama_priority(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.3")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        backend = get_llm_backend()
        assert isinstance(backend, OllamaBackend)

    def test_openrouter_priority(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fake")
        mock_openai = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        mock_openai.OpenAI = MagicMock()
        from tui.llm_backend import OpenRouterBackend
        backend = get_llm_backend()
        assert isinstance(backend, OpenRouterBackend)

    def test_openai_priority(self, clean_llm_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
        mock_openai = MagicMock()
        monkeypatch.setitem(sys.modules, "openai", mock_openai)
        mock_openai.OpenAI = MagicMock()
        from tui.llm_backend import OpenAIBackend
        backend = get_llm_backend()
        assert isinstance(backend, OpenAIBackend)

    def test_no_credentials_raises(self, clean_llm_env):
        with patch("tui.credentials.resolve_api_key", side_effect=RuntimeError("no key")):
            with pytest.raises(RuntimeError, match="No LLM backend configured"):
                get_llm_backend()


# ---------------------------------------------------------------------------
# ClaudeBackend.validate
# ---------------------------------------------------------------------------

class TestClaudeValidate:
    @patch("tui.credentials.resolve_api_key")
    def test_validate_success(self, mock_resolve, monkeypatch):
        from tui.credentials import CredentialSource
        mock_resolve.return_value = CredentialSource(api_key="sk-ant-fake", source="env")
        mock_anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        from tui.llm_backend import ClaudeBackend
        backend = ClaudeBackend()
        assert backend.validate() is True

    @patch("tui.credentials.resolve_api_key")
    def test_validate_auth_error(self, mock_resolve, monkeypatch):
        from tui.credentials import CredentialSource
        mock_resolve.return_value = CredentialSource(api_key="sk-ant-bad", source="env")
        mock_anthropic = MagicMock()
        # Create a real-looking AuthenticationError class
        mock_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        from tui.llm_backend import ClaudeBackend
        backend = ClaudeBackend()
        # Make the API call raise AuthenticationError
        backend._client.messages.create.side_effect = mock_anthropic.AuthenticationError("bad key")
        assert backend.validate() is False


# ---------------------------------------------------------------------------
# ClaudeBackend.generate_experiment
# ---------------------------------------------------------------------------

class TestGenerateExperiment:
    @patch("tui.credentials.resolve_api_key")
    def test_claude_generate_parses_response(self, mock_resolve, monkeypatch):
        from tui.credentials import CredentialSource
        mock_resolve.return_value = CredentialSource(api_key="sk-ant-fake", source="env")
        mock_anthropic = MagicMock()
        monkeypatch.setitem(sys.modules, "anthropic", mock_anthropic)

        from tui.llm_backend import ClaudeBackend
        backend = ClaudeBackend()

        # Mock API response
        mock_content = MagicMock()
        mock_content.text = VALID_RESPONSE
        mock_response = MagicMock()
        mock_response.content = [mock_content]
        backend._client.messages.create.return_value = mock_response

        hw = {"chip_name": "Test", "memory_gb": 16, "gpu_cores": 64, "peak_tflops": 10.0}
        result = backend.generate_experiment(
            current_code="x = 1",
            results_history="",
            best_val_bpb=1.5,
            best_experiment="baseline",
            hw_info=hw,
        )
        assert isinstance(result, ExperimentProposal)
        assert result.description == "Increase MATRIX_LR from 0.04 to 0.06"
