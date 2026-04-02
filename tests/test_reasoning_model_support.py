"""Tests for reasoning model support in AzureOpenAIBackend.

Reasoning models (Kimi K2.5, o3, etc.) consume completion tokens on
chain-of-thought before producing the actual response content. These
tests verify that the backend handles this correctly.
"""

import sys
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Patch hardware detection before importing llm_backend
_FAKE_HW = {
    "chip_name": "Test GPU",
    "memory_gb": 16,
    "gpu_cores": 64,
    "peak_tflops": 10.0,
    "chip_tier": "test",
}
with patch("backends.get_hardware_info", return_value=_FAKE_HW):
    from tui.llm_backend import AzureOpenAIBackend


def _make_azure_backend(monkeypatch, deployment="gpt-4.1"):
    """Create an AzureOpenAIBackend with mocked credentials and client."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake.openai.azure.com")
    mock_openai = MagicMock()
    monkeypatch.setitem(sys.modules, "openai", mock_openai)
    backend = AzureOpenAIBackend(model=deployment)
    return backend


class TestReasoningModelDetection:
    def test_kimi_is_reasoning(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        assert backend._is_reasoning_model is True

    def test_o3_is_reasoning(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "o3")
        assert backend._is_reasoning_model is True

    def test_o3_mini_is_reasoning(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "o3-mini")
        assert backend._is_reasoning_model is True

    def test_gpt41_is_not_reasoning(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "gpt-4.1")
        assert backend._is_reasoning_model is False

    def test_sonnet_is_not_reasoning(self, monkeypatch):
        # Sonnet goes through ClaudeBackend, not Azure, but test the logic
        backend = _make_azure_backend(monkeypatch, "claude-sonnet-4-6")
        assert backend._is_reasoning_model is False


class TestReasoningModelTokenBudget:
    def test_standard_model_uses_2048(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "gpt-4.1")
        assert not backend._is_reasoning_model
        assert backend.STANDARD_MAX_TOKENS == 2048

    def test_reasoning_model_uses_32768(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        assert backend._is_reasoning_model
        assert backend.REASONING_MAX_TOKENS == 32768


class TestReasoningModelValidation:
    def test_validate_reasoning_model_uses_more_tokens(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        # Mock successful response with content
        mock_msg = MagicMock()
        mock_msg.content = "OK"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        backend._client.chat.completions.create.return_value = mock_response

        assert backend.validate() is True
        # Verify max_tokens was 512 (not 16)
        call_kwargs = backend._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens", call_kwargs[1].get("max_tokens")) == 512

    def test_validate_standard_model_uses_16_tokens(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "gpt-4.1")
        mock_msg = MagicMock()
        mock_msg.content = "OK"
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        backend._client.chat.completions.create.return_value = mock_response

        assert backend.validate() is True
        call_kwargs = backend._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens", call_kwargs[1].get("max_tokens")) == 16


class TestReasoningModelGenerate:
    def test_none_content_raises_valueerror(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        # Mock response with content=None (reasoning exhausted budget)
        mock_msg = MagicMock()
        mock_msg.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        backend._client.chat.completions.create.return_value = mock_response

        hw = {"chip_name": "Test", "memory_gb": 16, "gpu_cores": 64, "peak_tflops": 10.0}
        with pytest.raises(ValueError, match="empty content"):
            backend.generate_experiment(
                current_code="x = 1",
                results_history="",
                best_val_bpb=1.5,
                best_experiment="baseline",
                hw_info=hw,
            )

    def test_valid_content_parses(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        mock_msg = MagicMock()
        mock_msg.content = (
            "DESCRIPTION: Increase MATRIX_LR from 0.04 to 0.06\n"
            "REASONING: Higher learning rate may help.\n"
            "CODE:\nMATRIX_LR = 0.06"
        )
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        backend._client.chat.completions.create.return_value = mock_response

        hw = {"chip_name": "Test", "memory_gb": 16, "gpu_cores": 64, "peak_tflops": 10.0}
        result = backend.generate_experiment(
            current_code="x = 1",
            results_history="",
            best_val_bpb=1.5,
            best_experiment="baseline",
            hw_info=hw,
        )
        assert result.description == "Increase MATRIX_LR from 0.04 to 0.06"

    def test_reasoning_model_gets_higher_max_tokens(self, monkeypatch):
        backend = _make_azure_backend(monkeypatch, "Kimi-K2.5")
        mock_msg = MagicMock()
        mock_msg.content = (
            "DESCRIPTION: test\nREASONING: test\nCODE:\nx = 1"
        )
        mock_choice = MagicMock()
        mock_choice.message = mock_msg
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        backend._client.chat.completions.create.return_value = mock_response

        hw = {"chip_name": "Test", "memory_gb": 16, "gpu_cores": 64, "peak_tflops": 10.0}
        backend.generate_experiment(
            current_code="x = 1", results_history="",
            best_val_bpb=1.5, best_experiment="baseline", hw_info=hw,
        )
        call_kwargs = backend._client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("max_tokens", call_kwargs[1].get("max_tokens")) == 32768
