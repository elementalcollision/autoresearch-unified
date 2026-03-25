# Cross-Platform Overview

> **Unified Codebase v2** — All results on this page are from `autoresearch-unified` commit `dacda45`, the first production run of the merged repository. Both platforms run identical code, identical LLM (Claude Sonnet 4), and identical experiment protocols. Results are validated with no data loss.

## Environment Manifest

| Spec | RTX PRO 6000 Blackwell | MI300X |
|------|----------------------|--------|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition | AMD Instinct MI300X OAM |
| **VRAM** | 102 GB GDDR7 | 206 GB HBM3 (192 usable) |
| **Architecture** | Blackwell (sm_100) | CDNA3 (gfx942) |
| **Compute Units** | — | 304 CUs |
| **Peak bf16 FLOPS** | 380 TFLOPS | 1,307 TFLOPS |
| **Memory BW** | ~1,100 GB/s | ~5,300 GB/s |
| **TDP** | 250W | 750W |
| **Driver** | 570.195.03 | ROCm 6.1.0-82 |
| **CUDA / HIP** | CUDA 12.8 | HIP 6.3.42131 |
| **PyTorch** | 2.8.0.dev+cu128 | 2.8.0+rocm6.3 |
| **torch.compile** | Enabled (default mode) | DISABLED in v2 suite (Inductor crash on CDNA3 with PyTorch 2.8.0 — see [fix/rocm-torch-compile-mi300x](https://github.com/elementalcollision/autoresearch-unified/pull/) for remediation) |
| **RunPod Image** | pytorch:2.8.0-py3.11-cuda12.8.1 | pytorch:2.4.0-py3.10-rocm6.1.0 |
| **Container Disk** | 80 GB | 80 GB |
| **Cost** | $1.69/hr | $1.99/hr ($0.50 secure) |
| **Training Script** | platforms/cuda/train_cuda.py | platforms/rocm/train_rocm.py |
| **Codebase Commit** | dacda45 | dacda45 |
| **LLM Agent** | Claude Sonnet 4 (claude-sonnet-4-20250514) | Claude Sonnet 4 (claude-sonnet-4-20250514) |
| **Provider** | RunPod | RunPod |

## Best val_bpb by Dataset and GPU

| Dataset | RTX PRO 6000 | MI300X | Status |
|---------|-------------|--------|--------|
| **ClimbMix** | **1.057228** | 1.085843 | Complete (both) |
| FineWeb-Edu | _in progress_ | _in progress_ | Running |
| Cosmopedia-v2 | — | — | Queued |
| SlimPajama | — | — | Queued |
| FineWeb-Edu-High | — | — | Queued |
| FineWeb | — | — | Queued |
| GitHub-Code-Python | — | — | Queued |
| PubMed-Abstract | — | — | Queued |

Bold = best result for that dataset. Table updates as datasets complete.

## ClimbMix — Detailed Comparison

| Metric | RTX PRO 6000 | MI300X |
|--------|-------------|--------|
| **Experiments** | 79 | 80 |
| **Best val_bpb** | **1.057228** | 1.085843 |
| **Baseline val_bpb** | 1.077205 | 1.124305 |
| **Improvement over baseline** | 1.85% | 3.42% |
| **Keeps** | 13 (16.5%) | 26 (32.5%) |
| **Discards** | 61 (77.2%) | 52 (65.0%) |
| **Crashes** | 4 (5.1%) | 1 (1.3%) |
| **Best tok/sec** | 290,139 | 253,994 |
| **Best MFU** | 33.7% | 7.1% |
| **VRAM at best** | 17.1 GB (17% of 102) | 64.6 GB (34% of 192) |
| **Steps at best** | 1,329 | 582 |
| **Baseline steps** | 1,161 | 355 |
| **Baseline depth** | 10 | 12 |
| **Baseline tok/sec** | 253,497 | 154,896 |
| **Baseline MFU** | 28.3% | 7.9% |
| **Time per experiment** | ~5 min | ~5 min |
| **Total runtime** | ~7 hrs | ~7 hrs |
| **Cost** | ~$11.83 | ~$13.93 |

### Optimization Paths

**RTX PRO 6000** — LR-first, then architecture:
1. MATRIX_LR 0.04 → 0.028 (exp2-15): 0.5% improvement
2. WINDOW_PATTERN SSSL → SSLL (exp31): **1.0% breakthrough** + 15% throughput boost
3. MATRIX_LR retuning for SSLL 0.028 → 0.022 (exp32-33): 0.2%
4. WEIGHT_DECAY 0.2 → 0.15 (exp41): 0.06%
5. WARMDOWN_RATIO 0.5 → 0.7 (exp43-45): 0.03%
6. SCALAR_LR 0.5 → 0.35 (exp51): 0.01%
7. FINAL_LR_FRAC 0.0 → 0.1 (exp75): 0.01%

**MI300X** — Architecture-first (constrained by low MFU):
1. FINAL_LR_FRAC tuning (exp11-14): 0.2% improvement
2. WINDOW_PATTERN SSSL → SSLL → SLSL (exp18-19): 0.7%
3. ASPECT_RATIO 64 → 38 (exp21-28): **2.0% breakthrough** (smaller model = more steps)
4. WEIGHT_DECAY 0.2 → 0.08 (exp32-78): 0.2%
5. All 4 learning rates rebalanced (exp35-51): 0.1%
6. ADAM_BETAS tuning (exp61-67): 0.05%
7. WINDOW_PATTERN SLSL → LLSS (exp69): 0.1%

## Key Findings

### 1. torch.compile Is the Dominant Variable

The RTX PRO 6000 achieves better absolute val_bpb (1.057 vs 1.086) despite having 3.4x fewer theoretical FLOPS. The difference is **torch.compile**: enabled on CUDA (33.7% MFU), disabled on ROCm due to an Inductor backend crash (7.1% MFU). This 4.7x MFU gap means the RTX PRO 6000 effectively trains ~4.7x faster per step, allowing deeper models and more training steps per experiment.

> **Remediation in progress**: Branch `fix/rocm-torch-compile-mi300x` adds a tiered compile fallback chain (Inductor → aot_eager → eager) and recommends upgrading to PyTorch 2.9.1+rocm6.3 which is expected to fix the Inductor shape inference bug on CDNA3. Target MFU after fix: 15-25%.

### 2. Platform Constraints Drive Different Optimization Strategies

The LLM agent independently discovered different strategies on each platform:
- **RTX PRO 6000** (compute-rich): Focused on learning rate precision, then found a window pattern breakthrough. Kept the default depth=10 model.
- **MI300X** (compute-limited): Immediately sought more training steps by reducing ASPECT_RATIO from 64 to 38 — a 41% reduction in model size that enabled 64% more steps (355→582). This was the single biggest improvement on this platform.

Both platforms discovered that WINDOW_PATTERN changes (SSSL→SSLL, SLSL, LLSS) provide significant gains — this insight transfers across hardware.

### 3. Keep Rate Inversely Correlates with Baseline Quality

MI300X had a 32.5% keep rate vs RTX PRO 6000's 16.5%. This isn't because MI300X is "easier to optimize" — it's because the MI300X baseline was weaker (1.124 vs 1.077), leaving more room for improvement. The agent found 3.42% improvement on MI300X vs 1.85% on RTX PRO 6000, consistent with diminishing returns near the optimum.

### 4. VRAM Utilization Is Low on Both Platforms

RTX PRO 6000 uses 17% of its 102 GB VRAM; MI300X uses 34% of its 192 GB. Both GPUs are dramatically over-provisioned for this model size. Smaller, cheaper GPUs (e.g., RTX 4090 24GB, L40S 48GB) could likely match these results with the same training scripts.

### 5. Crash Patterns Differ by Platform

RTX PRO 6000 crashed 4 times (5.1%) — all on batch size changes (TOTAL_BATCH_SIZE and DEVICE_BATCH_SIZE modifications). MI300X crashed once (1.3%) — also a batch size change. Both platforms are sensitive to batch size configuration in the training script, suggesting the divisibility assertion (`TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0`) is too strict.

## Data Integrity

| Check | RTX PRO 6000 | MI300X |
|-------|-------------|--------|
| TSV rows match heartbeat total | 79/79 | 80/80 |
| baseline_sha consistent | dacda45 (all rows) | dacda45 (all rows) |
| _ensure_clean_baseline at start | Confirmed in log | Confirmed in log |
| No data loss | Verified | Verified |
| GitHub sync | Branches pushed | Branches pushed |

## Cost Analysis

| GPU | ClimbMix Cost | Experiments | Cost/Experiment | Best val_bpb | Improvement |
|-----|--------------|-------------|-----------------|-------------|-------------|
| RTX PRO 6000 | ~$11.83 | 79 | $0.15 | 1.057228 | 1.85% |
| MI300X | ~$13.93 | 80 | $0.17 | 1.085843 | 3.42% |
| **Total** | **$25.76** | **159** | **$0.16** | | |

## GitHub Branches

| Branch | GPU | Dataset | Experiments | Status |
|--------|-----|---------|-------------|--------|
| `autoresearch/rtxpro6000-mar24-v2-climbmix` | RTX PRO 6000 | ClimbMix | 79 | Complete |
| `autoresearch/rtxpro6000-mar24-v2-fineweb-edu` | RTX PRO 6000 | FineWeb-Edu | In progress | Running |
| `autoresearch/mi300x-mar24-v2-climbmix` | MI300X | ClimbMix | 80 | Complete |
| `autoresearch/mi300x-mar24-v2-fineweb-edu` | MI300X | FineWeb-Edu | In progress | Running |

---

_Previous results from the pre-unification era (4 separate repos) are archived at [Archive: Cross-Platform Overview v1](Archive-Cross-Platform-Overview-v1)._
