# Dataset: ClimbMix

Source: [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) (10 shards, BPE tokenizer vocab_size=8192)

## Unified Codebase v2 Results (March 2026)

Codebase: `dacda45` | LLM: Claude Sonnet 4 | Protocol: 80 experiments, 5 min each

| GPU | Platform | Best val_bpb | Baseline | Improvement | Keeps | Crashes | Tok/sec | MFU | VRAM Used | Steps |
|-----|----------|-------------|----------|-------------|-------|---------|---------|-----|-----------|-------|
| **RTX PRO 6000 Blackwell** | CUDA 12.8 | **1.057228** | 1.077205 | 1.85% | 13 (16.5%) | 4 (5.1%) | 290,139 | 33.7% | 17.1 GB | 1,329 |
| MI300X | ROCm 6.3 | 1.085843 | 1.124305 | 3.42% | 26 (32.5%) | 1 (1.3%) | 253,994 | 7.1% | 64.6 GB | 582 |

### Key Optimizations Found

**RTX PRO 6000** (13 keeps across 79 experiments):
- exp75 (best): FINAL_LR_FRAC 0.0 → 0.1 (val_bpb 1.057228)
- exp31: WINDOW_PATTERN SSSL → SSLL (biggest single gain: 1.072 → 1.063)
- exp7-15: MATRIX_LR 0.04 → 0.028 (steady LR tuning)
- exp33: MATRIX_LR retuned to 0.022 for SSLL pattern
- exp41-45: WEIGHT_DECAY 0.2 → 0.15, WARMDOWN_RATIO 0.5 → 0.7

**MI300X** (26 keeps across 80 experiments):
- exp78 (best): WEIGHT_DECAY 0.1 → 0.08 (val_bpb 1.085843)
- exp21-28: ASPECT_RATIO 64 → 38 (biggest single gain: 1.117 → 1.090, more steps in 5 min)
- exp18-19: WINDOW_PATTERN SSSL → SSLL → SLSL
- exp69: WINDOW_PATTERN → LLSS
- exp32-33: WEIGHT_DECAY 0.2 → 0.1
- exp61-67: ADAM_BETAS tuning (0.8 → 0.65, 0.95 → 0.90)

### GitHub Branches
- RTX PRO 6000: [`autoresearch/rtxpro6000-mar24-v2-climbmix`](https://github.com/elementalcollision/autoresearch-unified/tree/autoresearch/rtxpro6000-mar24-v2-climbmix)
- MI300X: [`autoresearch/mi300x-mar24-v2-climbmix`](https://github.com/elementalcollision/autoresearch-unified/tree/autoresearch/mi300x-mar24-v2-climbmix)

---

## v1 Results (Pre-Unification Archive)

> Results below are from the old separate repositories which had known data loss bugs.

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| MI300X | amd_rocm | runpod | 1.0359 | 97 | 16.5% | 0.0% |
| A100 40GB | nvidia_cuda | vultr | 1.1422 | 94 | 24.5% | 0.0% |
| M5 Max | apple_metal | local | 1.2959 | 204 | 3.9% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.4468 | 50 | 12.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "climbmix") & (df["val_bpb"] > 0)]
```
