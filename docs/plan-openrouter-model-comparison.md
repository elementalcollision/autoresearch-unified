# Plan: Legacy Cleanup + OpenRouter Model Comparison Testing

## Context

We're consolidating all autoresearch work onto `autoresearch-unified` and starting cross-model comparison testing via OpenRouter. Multiple legacy directories (CUDA-only, ARM, macOS, MLX, ROCm, Gaudi forks) are no longer needed. The unified repo already has full OpenRouter backend support — we just need to add multi-model orchestration and clean up the old directories.

---

## Part 1: Legacy Directory Cleanup

### 1a. Archive valuable experiment data

Copy results TSVs from legacy repos into `autoresearch-unified/results/archive/`:

```
results/archive/cuda-legacy/        <- autoresearch-cuda/results/
results/archive/multi-dataset/      <- multi-dataset/autoresearch/results/
results/archive/arm-legacy/         <- autoresearch_ARM/results/
```

### 1b. Push all legacy repos (preserve remote state)

For each repo with untracked changes, commit and push a final snapshot before deletion:
- `autoresearch_ARM` (untracked results, run.log)
- `autoresearch-cuda` (untracked results)
- `ROCm` (untracked analysis.ipynb)
- `multi-dataset/autoresearch` (untracked results)

### 1c. Delete legacy directories

| Directory | Est. Size | Notes |
|-----------|-----------|-------|
| `autoresearch-macos` | ~1 MB | Third-party fork, no local changes |
| `autoresearch-mlx` | ~400 KB | Third-party fork, no local changes |
| `autoresearch_ARM` | ~830 MB | Has .venv; archive results first |
| `framework_autoresearch` | ~800 MB | Has .venv; no unique results |
| `autoresearch-cuda` | ~5 MB | Archive results first |
| `autoresearch-cuda.wiki` | ~2 MB | Wiki markdown |
| `Gaudi3` | ~1 MB | Old Gaudi-specific fork |
| `ROCm` | ~208 MB | Push untracked first |
| `multi-dataset/` | ~364 MB | Archive results first |

**Estimated disk recovery: ~2.2 GB**

---

## Part 2: OpenRouter Model Comparison Setup

### 2a. Model Tier Matrix

Add a `MODEL_TIERS` constant to `run_suite.py` (after `DEFAULT_MODEL`):

| Tier | Anthropic Baseline | OpenAI | Qwen | DeepSeek |
|------|-------------------|--------|------|----------|
| **Haiku-class** | `anthropic/claude-haiku-4-5` | `openai/gpt-4.1-mini` | `qwen/qwen-2.5-72b-instruct` | `deepseek/deepseek-chat-v3-0324` |
| **Sonnet-class** | `anthropic/claude-sonnet-4` | `openai/gpt-4.1` | `qwen/qwen-2.5-72b-instruct` | `deepseek/deepseek-chat-v3-0324` |
| **Opus-class** | `anthropic/claude-opus-4` | `openai/o3` | `qwen/qwen-2.5-72b-instruct` | `deepseek/deepseek-reasoner` |

> Note: Using `qwen/qwen-2.5-72b-instruct` across all tiers as the Qwen representative. Using only `openai/o3` (not gpt-4.1) at the Opus tier.

### 2b. Add `--models` flag to `run_suite.py`

- Add `MODEL_TIERS` dict after `DEFAULT_MODEL`
- Add `--models` CLI argument (accepts tier name like `haiku` or comma-separated model IDs)
- Wrap the existing dataset loop in an outer model loop so each model runs the same dataset(s) sequentially

### 2c. Results directory structure (already supported)

The existing `_model_slug()` function handles OpenRouter model IDs correctly:

```
results/
  climbmix/results.tsv                        # default Sonnet
  haiku-4-5/climbmix/results.tsv              # anthropic/claude-haiku-4-5
  gpt-4.1-mini/climbmix/results.tsv           # openai/gpt-4.1-mini
  qwen-2.5-72b-instruct/climbmix/results.tsv  # qwen/qwen-2.5-72b-instruct
  deepseek-chat-v3-0324/climbmix/results.tsv  # deepseek/deepseek-chat-v3-0324
  ...
```

### 2d. Create `compare_models.py`

New script modeled on existing `compare_datasets.py` that:
- Scans `results/` for model-slug subdirectories
- For a given dataset, loads best val_bpb from each model
- Generates comparison table grouped by tier
- Shows keep rate, experiments-to-best, and improvement trajectory per model

---

## Part 3: Critical Files

| File | Change |
|------|--------|
| `run_suite.py` | Add `MODEL_TIERS` dict, `--models` arg, outer model loop |
| `compare_models.py` | **New** — cross-model comparison tool |
| `tui/llm_backend.py` | No changes needed (OpenRouter backend already works) |

---

## Part 4: Testing / Verification

1. **Smoke test** — After adding `--models`, run 5 experiments each on one model per provider against `climbmix` to confirm connectivity and response parsing:
   ```
   uv run run_suite.py --dataset climbmix --max-experiments 5 --model "openai/gpt-4.1-mini"
   uv run run_suite.py --dataset climbmix --max-experiments 5 --model "qwen/qwen-2.5-72b-instruct"
   uv run run_suite.py --dataset climbmix --max-experiments 5 --model "deepseek/deepseek-chat-v3-0324"
   ```
2. **Tier run** — Full 80-experiment comparison:
   ```
   uv run run_suite.py --dataset climbmix --max-experiments 80 --models haiku
   ```
3. **Compare** — Run `compare_models.py --dataset climbmix` to see cross-model results

### Known risks
- Non-Anthropic models may not consistently follow the `DESCRIPTION:/REASONING:/CODE:` response format — the parser uses `re.MULTILINE` so preamble is OK, but completely different formats will crash (handled by orchestrator as experiment crash)
- Reasoning models (o3, deepseek-reasoner) emit chain-of-thought before structured output — test with 5 experiments first
- OpenRouter rate limits vary by model — existing backoff in orchestrator should handle this
