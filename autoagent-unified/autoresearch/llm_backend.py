# Ported from autoresearch-unified/tui/llm_backend.py (MIT)
"""LLM backend abstraction for generating experiment modifications.

Supports Claude API, OpenAI, Azure OpenAI, OpenRouter, and Ollama (placeholder).
System prompt adapts dynamically to detected hardware and platform.
"""

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExperimentProposal:
    """A proposed code modification from the LLM."""
    description: str  # one-line description for results.tsv
    reasoning: str    # 2-3 sentences explaining the hypothesis
    code: str         # replacement code for the hyperparameter block


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def get_system_prompt(hw_info=None):
    """Generate system prompt tailored to the detected hardware and platform."""
    if hw_info is None:
        from backends import get_hardware_info
        hw_info = get_hardware_info()

    gpu_name = hw_info.get("chip_name", "GPU")
    mem_gb = hw_info.get("memory_gb", 0)
    tier = hw_info.get("chip_tier", "unknown")

    rocm_ver = hw_info.get("rocm_version")
    if rocm_ver:
        if rocm_ver[0] >= 7:
            platform_str = f"AMD GPU with ROCm {rocm_ver[0]}.{rocm_ver[1]}"
            platform_notes = (
                "torch.compile (reduce-overhead mode) uses HIP graph capture, "
                "CK Flash Attention is explicitly selected, bf16 autocast is on."
            )
        else:
            platform_str = f"AMD GPU with ROCm {rocm_ver[0]}.{rocm_ver[1]}"
            platform_notes = (
                "torch.compile (default mode) fuses kernels via AMD Triton, "
                "SDPA dispatches to CK-based attention, bf16 autocast is on."
            )
    elif tier == "gaudi3":
        platform_str = "Intel Gaudi 3 HPU"
        platform_notes = (
            "Gaudi 3 has 128 GB HBM2e and ~1835 TFLOPS bf16 per device. "
            "torch.compile uses hpu_backend, FusedSDPA for attention, bf16 autocast is on."
        )
    elif any(x in gpu_name.lower() for x in ["nvidia", "geforce", "rtx", "tesla", "h100", "a100"]):
        platform_str = f"NVIDIA GPU ({gpu_name})"
        platform_notes = (
            "torch.compile with architecture-specific inductor config, "
            "activation checkpointing on supported GPUs, bf16 autocast is on."
        )
    else:
        platform_str = f"Apple Silicon ({gpu_name})"
        platform_notes = (
            "MLX-native training with MuonAdamW optimizer. "
            "Unified memory shared between CPU and GPU."
        )

    memory_note = (
        f"HBM ({mem_gb:.0f}GB) is generous -- the time budget is the dominant constraint, not memory."
        if mem_gb >= 40
        else f"Memory ({mem_gb:.0f}GB) is limited -- be mindful of depth and batch sizes."
    )

    return f"""\
You are an autonomous AI researcher optimizing a small language model on {platform_str}.

You modify the hyperparameter block of a training script to minimize val_bpb (validation bits per byte -- lower is better). Each experiment runs for a fixed 5-minute time budget.

Rules:
- You may ONLY modify the hyperparameter block shown between the marker comments.
- You may change: batch sizes, learning rates, weight decay, warmup/warmdown ratios, model depth, aspect ratio, head dim, window pattern, MLP ratio, or any constant in that block.
- You may NOT add imports, modify the model class, optimizer, data loading, or evaluation.
- Make ONE change per experiment. This isolates the effect and makes results interpretable.
- Consider the full results history -- don't repeat failed experiments.
- If many experiments have been discarded, try a different direction entirely.
- Platform-specific: {platform_notes} {memory_note}

Strategy guidance (use the full repertoire, not just learning rate tuning):
- Learning rates: MATRIX_LR, SCALAR_LR, EMBEDDING_LR, UNEMBEDDING_LR
- Regularization: WEIGHT_DECAY, ADAM_BETAS (beta1, beta2)
- Schedule shape: WARMUP_RATIO (try >0), WARMDOWN_RATIO, FINAL_LR_FRAC (try >0)
- Architecture: DEPTH, ASPECT_RATIO, HEAD_DIM, MLP_RATIO, WINDOW_PATTERN
- Throughput: TOTAL_BATCH_SIZE, DEVICE_BATCH_SIZE (halving batch = 2x steps = often big wins)
- Untried levers: ACTIVATION_CHECKPOINTING (enables deeper models), COMPILE_MODE

Respond in EXACTLY this format (no markdown fences around the whole response):

DESCRIPTION: <one-line description, e.g. "Increase MATRIX_LR from 0.04 to 0.06">
REASONING: <2-3 sentences explaining why this change might improve val_bpb>
CODE:
<the complete replacement hyperparameter block, from the opening marker comment to the closing marker comment, inclusive>
"""


USER_PROMPT_TEMPLATE = """\
Here is the current hyperparameter block of the training script:

```python
{current_code}
```

Here is the experiment history so far:

{results_history}

Hardware: {chip_name}, {memory_gb:.0f} GB, {gpu_cores} cores/SMs/TPCs, ~{peak_tflops:.1f} TFLOPS bf16

Current best val_bpb: {best_val_bpb:.6f} (from {best_experiment})

Propose the next experiment. Remember: ONE change, and respond in the exact format specified.
"""


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> ExperimentProposal:
    """Parse the LLM response into an ExperimentProposal."""
    desc_match = re.search(r'^DESCRIPTION:\s*(.+?)$', response_text, re.MULTILINE)
    if not desc_match:
        raise ValueError("Response missing DESCRIPTION field")
    description = desc_match.group(1).strip()

    reason_match = re.search(r'^REASONING:\s*(.+?)(?=\nCODE:)', response_text, re.MULTILINE | re.DOTALL)
    if not reason_match:
        raise ValueError("Response missing REASONING field")
    reasoning = reason_match.group(1).strip()

    code_match = re.search(r'^CODE:\s*\n(.*)', response_text, re.MULTILINE | re.DOTALL)
    if not code_match:
        raise ValueError("Response missing CODE field")
    code = code_match.group(1).strip()

    code = re.sub(r'^```(?:python)?\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)

    return ExperimentProposal(
        description=description,
        reasoning=reasoning,
        code=code,
    )


# ---------------------------------------------------------------------------
# Backend ABC
# ---------------------------------------------------------------------------

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        ...

    @abstractmethod
    def name(self) -> str:
        ...

    def validate(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Claude Backend
# ---------------------------------------------------------------------------

class ClaudeBackend(LLMBackend):
    """Claude API backend using the Anthropic SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        from autoresearch.credentials import resolve_api_key

        cred = resolve_api_key()
        self._client = anthropic.Anthropic(api_key=cred.api_key)
        self._model = model or os.environ.get("CLAUDE_MODEL") or self.DEFAULT_MODEL
        self._cred_source = cred.source

    def name(self) -> str:
        return f"Claude ({self._model}) via {self._cred_source}"

    def validate(self) -> bool:
        try:
            import anthropic
            self._client.messages.create(
                model=self._model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except anthropic.AuthenticationError:
            return False

    def generate_experiment(self, current_code, results_history, best_val_bpb, best_experiment, hw_info):
        system_prompt = get_system_prompt(hw_info)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history or "No experiments yet.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return parse_llm_response(response.content[0].text)


# ---------------------------------------------------------------------------
# OpenAI Backend
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """OpenAI API backend."""

    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL

    def name(self) -> str:
        return f"OpenAI ({self._model})"

    def validate(self) -> bool:
        try:
            from openai import AuthenticationError
            self._client.chat.completions.create(
                model=self._model, max_tokens=16,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except AuthenticationError:
            return False

    def generate_experiment(self, current_code, results_history, best_val_bpb, best_experiment, hw_info):
        system_prompt = get_system_prompt(hw_info)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history or "No experiments yet.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.chat.completions.create(
            model=self._model, max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return parse_llm_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Azure OpenAI Backend
# ---------------------------------------------------------------------------

class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI Service backend with reasoning model support."""

    DEFAULT_DEPLOYMENT = "gpt-4.1"
    DEFAULT_API_VERSION = "2024-12-01-preview"
    REASONING_MODELS = {"kimi-k2.5", "o3", "o3-mini", "o4-mini"}
    REASONING_MAX_TOKENS = 32768
    STANDARD_MAX_TOKENS = 2048

    def __init__(self, model: str | None = None):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY environment variable not set")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT environment variable not set")

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or self.DEFAULT_API_VERSION
        self._client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        self._deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or self.DEFAULT_DEPLOYMENT
        self._endpoint = endpoint

    def name(self) -> str:
        return f"Azure OpenAI ({self._deployment}) via {self._endpoint}"

    @property
    def _is_reasoning_model(self) -> bool:
        return self._deployment.lower() in self.REASONING_MODELS

    def validate(self) -> bool:
        try:
            from openai import AuthenticationError
            max_tokens = 512 if self._is_reasoning_model else 16
            self._client.chat.completions.create(
                model=self._deployment, max_tokens=max_tokens,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except AuthenticationError:
            return False

    def generate_experiment(self, current_code, results_history, best_val_bpb, best_experiment, hw_info):
        system_prompt = get_system_prompt(hw_info)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history or "No experiments yet.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        max_tokens = self.REASONING_MAX_TOKENS if self._is_reasoning_model else self.STANDARD_MAX_TOKENS
        response = self._client.chat.completions.create(
            model=self._deployment, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        response_text = response.choices[0].message.content
        if response_text is None:
            raise ValueError(f"LLM returned empty content (model={self._deployment})")
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# OpenRouter Backend
# ---------------------------------------------------------------------------

class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend — unified access to 200+ models."""

    DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/elementalcollision/autoagent-unified",
                "X-Title": "autoagent-unified",
            },
        )
        self._model = model or os.environ.get("OPENROUTER_MODEL") or self.DEFAULT_MODEL

    def name(self) -> str:
        return f"OpenRouter ({self._model})"

    def validate(self) -> bool:
        try:
            from openai import AuthenticationError
            self._client.chat.completions.create(
                model=self._model, max_tokens=16,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except AuthenticationError:
            return False

    def generate_experiment(self, current_code, results_history, best_val_bpb, best_experiment, hw_info):
        system_prompt = get_system_prompt(hw_info)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history or "No experiments yet.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.chat.completions.create(
            model=self._model, max_tokens=2048,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return parse_llm_response(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Ollama Backend (placeholder)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """Local LLM backend via Ollama (placeholder)."""

    def __init__(self):
        self._model = os.environ.get("OLLAMA_MODEL", "")

    def name(self) -> str:
        return f"Ollama ({self._model})"

    def generate_experiment(self, current_code, results_history, best_val_bpb, best_experiment, hw_info):
        raise NotImplementedError(f"Ollama backend ({self._model}) is not yet implemented.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_backend(model: str | None = None) -> LLMBackend:
    """Create the appropriate LLM backend based on available credentials.

    Priority: Ollama > OpenRouter > Azure OpenAI > OpenAI > Claude
    """
    if os.environ.get("OLLAMA_MODEL"):
        return OllamaBackend()

    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterBackend(model=model)

    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIBackend(model=model)

    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIBackend(model=model)

    try:
        return ClaudeBackend(model=model)
    except (RuntimeError, ImportError):
        pass

    raise RuntimeError(
        "No LLM backend configured. Set up credentials for one of:\n"
        "\n"
        "  Anthropic Claude:  export ANTHROPIC_API_KEY=sk-ant-...\n"
        "  OpenRouter:        export OPENROUTER_API_KEY=sk-or-...\n"
        "  OpenAI:            export OPENAI_API_KEY=sk-...\n"
        "  Azure OpenAI:      export AZURE_OPENAI_API_KEY=... + AZURE_OPENAI_ENDPOINT=...\n"
        "  Ollama (local):    export OLLAMA_MODEL=llama3.3\n"
    )
