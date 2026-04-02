"""LLM backend abstraction for generating experiment modifications.

Supports Claude API (Option A) with a placeholder for local LLMs via Ollama (Option B).
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

    # Determine platform-specific notes
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
- The key insight from prior characterization: maximizing gradient steps within the fixed time budget is the dominant factor. Smaller batches = more steps = usually better, up to a point.
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


SYSTEM_PROMPT = get_system_prompt()

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
    """Parse the LLM response into an ExperimentProposal.

    Expected format:
        DESCRIPTION: ...
        REASONING: ...
        CODE:
        <code block>
    """
    # Extract DESCRIPTION
    desc_match = re.search(r'^DESCRIPTION:\s*(.+?)$', response_text, re.MULTILINE)
    if not desc_match:
        raise ValueError("Response missing DESCRIPTION field")
    description = desc_match.group(1).strip()

    # Extract REASONING
    reason_match = re.search(r'^REASONING:\s*(.+?)(?=\nCODE:)', response_text, re.MULTILINE | re.DOTALL)
    if not reason_match:
        raise ValueError("Response missing REASONING field")
    reasoning = reason_match.group(1).strip()

    # Extract CODE -- everything after "CODE:" line
    code_match = re.search(r'^CODE:\s*\n(.*)', response_text, re.MULTILINE | re.DOTALL)
    if not code_match:
        raise ValueError("Response missing CODE field")
    code = code_match.group(1).strip()

    # Strip markdown code fences if present
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
        """Generate a proposed code modification."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
        ...

    def validate(self) -> bool:
        """Test that the backend credentials work. Returns True if valid."""
        return True  # Default: assume valid


# ---------------------------------------------------------------------------
# Claude Backend (Option A)
# ---------------------------------------------------------------------------

class ClaudeBackend(LLMBackend):
    """Claude API backend using the Anthropic SDK."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, model: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )

        from tui.credentials import resolve_api_key

        cred = resolve_api_key()
        self._client = anthropic.Anthropic(api_key=cred.api_key)
        self._model = model or os.environ.get("CLAUDE_MODEL") or self.DEFAULT_MODEL
        self._cred_source = cred.source

    def name(self) -> str:
        return f"Claude ({self._model}) via {self._cred_source}"

    def validate(self) -> bool:
        """Test the API key with a minimal request. Returns True if valid."""
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

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history if results_history else "No experiments yet -- this will be the first modification after baseline.",
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
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        response_text = response.content[0].text
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# OpenAI Backend
# ---------------------------------------------------------------------------

class OpenAIBackend(LLMBackend):
    """OpenAI API backend (GPT-4.1, GPT-5.1, etc.)."""

    DEFAULT_MODEL = "gpt-4.1"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self._client = OpenAI(api_key=api_key)
        self._model = model or os.environ.get("OPENAI_MODEL") or self.DEFAULT_MODEL

    def name(self) -> str:
        return f"OpenAI ({self._model})"

    def validate(self) -> bool:
        """Test the API key with a minimal request."""
        try:
            from openai import AuthenticationError
            self._client.chat.completions.create(
                model=self._model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except AuthenticationError:
            return False

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history if results_history else "No experiments yet -- this will be the first modification after baseline.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# Azure OpenAI Backend
# ---------------------------------------------------------------------------

class AzureOpenAIBackend(LLMBackend):
    """Azure OpenAI Service backend.

    Supports both standard models (GPT-4.1) and reasoning models
    (Kimi K2.5, o3) that use chain-of-thought tokens. Reasoning models
    need a higher max_completion_tokens budget because they consume
    tokens on hidden reasoning before producing the actual response.

    Requires:
        AZURE_OPENAI_API_KEY     - API key for the Azure resource
        AZURE_OPENAI_ENDPOINT    - Resource endpoint (e.g. https://my-resource.openai.azure.com)
        AZURE_OPENAI_DEPLOYMENT  - Deployment name (default: gpt-4.1)
        AZURE_OPENAI_API_VERSION - API version (default: 2024-12-01-preview)
    """

    DEFAULT_DEPLOYMENT = "gpt-4.1"
    DEFAULT_API_VERSION = "2024-12-01-preview"

    # Reasoning models burn completion tokens on chain-of-thought before
    # producing the actual response content. With a low budget, the model
    # exhausts all tokens on reasoning and returns content=None.
    REASONING_MODELS = {"kimi-k2.5", "o3", "o3-mini", "o4-mini"}
    REASONING_MAX_TOKENS = 32768
    STANDARD_MAX_TOKENS = 2048

    def __init__(self, model: str | None = None):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY environment variable not set")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT environment variable not set")

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION") or self.DEFAULT_API_VERSION
        self._client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        self._deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT") or self.DEFAULT_DEPLOYMENT
        self._endpoint = endpoint

    def name(self) -> str:
        return f"Azure OpenAI ({self._deployment}) via {self._endpoint}"

    @property
    def _is_reasoning_model(self) -> bool:
        return self._deployment.lower() in self.REASONING_MODELS

    def validate(self) -> bool:
        """Test the API key and deployment with a minimal request.

        Reasoning models need a higher token budget even for validation
        because chain-of-thought consumes tokens before producing content.
        """
        try:
            from openai import AuthenticationError
            # Reasoning models need ~500 tokens to produce "OK" after thinking.
            max_tokens = 512 if self._is_reasoning_model else 16
            response = self._client.chat.completions.create(
                model=self._deployment,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            content = response.choices[0].message.content
            # For reasoning models, content can be None if the token budget
            # was still too small. Treat as valid (auth worked) but warn.
            if content is None and self._is_reasoning_model:
                return True  # Auth is fine; generate_experiment uses more tokens
            return True
        except AuthenticationError:
            return False

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history if results_history else "No experiments yet -- this will be the first modification after baseline.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        max_tokens = self.REASONING_MAX_TOKENS if self._is_reasoning_model else self.STANDARD_MAX_TOKENS
        response = self._client.chat.completions.create(
            model=self._deployment,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content
        if response_text is None:
            raise ValueError(
                f"LLM returned empty content (model={self._deployment}, "
                f"max_tokens={max_tokens}). "
                f"Reasoning model may need a higher token budget."
            )
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# OpenRouter Backend
# ---------------------------------------------------------------------------

class OpenRouterBackend(LLMBackend):
    """OpenRouter API backend — unified access to 200+ models via OpenAI-compatible API.

    Uses the same openai SDK with a custom base_url. One API key provides
    access to Claude, GPT, Gemini, Llama, Mistral, and more.

    Env vars:
        OPENROUTER_API_KEY: API key from openrouter.ai (required)
        OPENROUTER_MODEL:   Model ID in provider/model format (default: anthropic/claude-sonnet-4.6)
    """

    DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"

    def __init__(self, model: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

        self._client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/elementalcollision/autoresearch-unified",
                "X-Title": "autoresearch-unified",
            },
        )
        self._model = model or os.environ.get("OPENROUTER_MODEL") or self.DEFAULT_MODEL

    def name(self) -> str:
        return f"OpenRouter ({self._model})"

    def validate(self) -> bool:
        """Test the API key with a minimal request."""
        try:
            from openai import AuthenticationError
            self._client.chat.completions.create(
                model=self._model,
                max_tokens=16,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True
        except AuthenticationError:
            return False

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        user_prompt = USER_PROMPT_TEMPLATE.format(
            current_code=current_code,
            results_history=results_history if results_history else "No experiments yet -- this will be the first modification after baseline.",
            chip_name=hw_info.get("chip_name", "Unknown"),
            memory_gb=hw_info.get("memory_gb", 0),
            gpu_cores=hw_info.get("gpu_cores", 0),
            peak_tflops=hw_info.get("peak_tflops", 0),
            best_val_bpb=best_val_bpb,
            best_experiment=best_experiment,
        )

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        response_text = response.choices[0].message.content
        return parse_llm_response(response_text)


# ---------------------------------------------------------------------------
# Ollama Backend (placeholder)
# ---------------------------------------------------------------------------

class OllamaBackend(LLMBackend):
    """Local LLM backend via Ollama (placeholder for future implementation)."""

    def __init__(self):
        self._model = os.environ.get("OLLAMA_MODEL", "")

    def name(self) -> str:
        return f"Ollama ({self._model})"

    def generate_experiment(
        self,
        current_code: str,
        results_history: str,
        best_val_bpb: float,
        best_experiment: str,
        hw_info: dict,
    ) -> ExperimentProposal:
        raise NotImplementedError(
            f"Ollama backend ({self._model}) is not yet implemented. "
            "Set ANTHROPIC_API_KEY to use Claude, or contribute an Ollama "
            "implementation to tui/llm_backend.py."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_backend(model: str | None = None) -> LLMBackend:
    """Create the appropriate LLM backend based on available credentials.

    Priority:
    1. OLLAMA_MODEL env var -> OllamaBackend (local, placeholder)
    2. OPENROUTER_API_KEY -> OpenRouterBackend (unified multi-model router)
    3. AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT -> AzureOpenAIBackend
    4. OPENAI_API_KEY -> OpenAIBackend
    5. Anthropic credentials (env/keychain/file) -> ClaudeBackend
    6. Error with setup instructions for all providers
    """
    # 1. Ollama (local)
    if os.environ.get("OLLAMA_MODEL"):
        return OllamaBackend()

    # 2. OpenRouter (unified router — before direct providers because if
    # someone sets OPENROUTER_API_KEY, they want traffic routed through it)
    if os.environ.get("OPENROUTER_API_KEY"):
        return OpenRouterBackend(model=model)

    # 3. Azure OpenAI (check before OpenAI — more specific env vars)
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIBackend(model=model)

    # 4. OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIBackend(model=model)

    # 5. Anthropic Claude (default)
    try:
        return ClaudeBackend(model=model)
    except (RuntimeError, ImportError):
        pass

    raise RuntimeError(
        "No LLM backend configured. Set up credentials for one of:\n"
        "\n"
        "  Anthropic Claude (recommended):\n"
        "    export ANTHROPIC_API_KEY=sk-ant-...\n"
        "\n"
        "  OpenRouter (200+ models, one API key):\n"
        "    export OPENROUTER_API_KEY=sk-or-...\n"
        "    export OPENROUTER_MODEL=anthropic/claude-sonnet-4.6  # optional\n"
        "\n"
        "  OpenAI:\n"
        "    export OPENAI_API_KEY=sk-...\n"
        "\n"
        "  Azure OpenAI:\n"
        "    export AZURE_OPENAI_API_KEY=...\n"
        "    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com\n"
        "\n"
        "  Ollama (local, experimental):\n"
        "    export OLLAMA_MODEL=llama3.3\n"
    )
