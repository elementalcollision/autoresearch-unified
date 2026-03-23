# Platform: AMD ROCm

AMD ROCm platform running on a single MI300X (192 GB HBM3) via RunPod cloud. Uses `torch.compile` with AMD Triton and CK-based FlashAttention for autonomous LLM-driven hyperparameter optimization.

## Summary

| Stat | Value |
|------|-------|
| **Platform** | `amd_rocm` |
| **Total experiments** | 322 |
| **GPU** | AMD Instinct MI300X (192 GB HBM3, CDNA 3) |
| **Datasets** | 4 (ClimbMix, FineWeb-Edu, Cosmopedia-v2, SlimPajama) |
| **Provider** | RunPod (~$1.99/hr) |
| **Architecture** | CDNA 3 (gfx942), 304 Compute Units |

## Software Stack

| Component | Version |
|-----------|---------|
| **ROCm** | 6.3 (HIP 6.3.42134) |
| **PyTorch** | 2.9.1+rocm6.3 |
| **Triton** | 3.5.1 (pytorch-triton-rocm) |
| **Python** | 3.10.14 |
| **OS** | Ubuntu 22.04.4 LTS |
| **Compilation** | `torch.compile(mode="default")` -- AMD Triton backend |
| **Attention** | SDPA dispatching to CK-based (Composable Kernel) FlashAttention |
| **Precision** | bf16 autocast on CDNA 3 matrix cores |

### ROCm 6.x vs 7.x

Two training scripts are maintained. ROCm 6.x is frozen during active data collection; 7.x adds HIP graph capture and explicit CK backend selection.

| Feature | ROCm 6.x (`train_rocm.py`) | ROCm 7.x (`train_rocm7.py`) |
|---------|---------------------------|---------------------------|
| torch.compile | `mode="default"` | `mode="reduce-overhead"` (HIP graphs) |
| Flash Attention | Auto-selected by SDPA | CK explicitly via `preferred_rocm_fa_library("ck")` |
| Status | **Production** (active data collection) | Ready for validation |

## Results by Dataset

| Dataset | Best val_bpb | Baseline | Improvement | Experiments | tok/s | MFU | Page |
|---------|-------------|----------|-------------|-------------|-------|-----|------|
| **ClimbMix** | **1.036** | 1.067 | -2.91% | 97 | 397K | 13.1% | [Dataset: ClimbMix](Dataset-Climbmix) |
| **FineWeb-Edu** | **1.015** | 1.036 | -2.05% | 99 | 550K | 18.3% | [Dataset: FineWeb-Edu](Dataset-Fineweb-Edu) |
| **Cosmopedia-v2** | 1.015 | — | — | 99 | — | — | [Dataset: Cosmopedia-v2](Dataset-Cosmopedia-V2) |
| **SlimPajama** | 1.015 | — | — | 22 | — | — | [Dataset: SlimPajama](Dataset-Slimpajama) |

The MI300X holds the **best absolute val_bpb** on ClimbMix (1.036), FineWeb-Edu (1.015), and SlimPajama (1.015) across all platforms.

## Key Finding: The Depth-Steps Tradeoff

Reducing depth from 12 to 10 on MI300X yielded **50% more training steps** in the 5-minute budget (880 to 911 steps) and dropped peak memory from 47.6 to 33.6 GB. With a fixed time budget, **more steps beat more parameters**.

This is the inverse of the CUDA finding where VRAM enables larger models. On MI300X, memory is never the bottleneck (192 GB available, ~34 GB used). Instead, the 5-minute wall-clock limit means throughput (steps/minute) is the binding constraint, and shallower models train faster.

## MFU Analysis

The MI300X's MFU is notably low relative to its theoretical 1,307 bf16 TFLOPS:

| Dataset | MFU | tok/s |
|---------|-----|-------|
| ClimbMix | 13.1% | 397K |
| FineWeb-Edu | 18.3% | 550K |

The ~13-18% MFU reflects the small model size relative to the MI300X's massive compute capacity. The workload is memory-bandwidth-bound rather than compute-bound at this scale. Despite low utilization, the MI300X's raw throughput (~400-550K tok/s) still exceeds all NVIDIA GPUs except the A100 on specific datasets, enabling more training steps and lower absolute val_bpb.

## Hardware Specifications

| Spec | MI300X |
|------|--------|
| **Architecture** | CDNA 3 (gfx942) |
| **Memory** | 192 GB HBM3 |
| **Memory bandwidth** | 5,300 GB/s |
| **bf16 TFLOPS** | 1,307 |
| **Compute Units** | 304 |
| **TDP** | 750W |
| **Cost/hr** | ~$1.99 (RunPod) |

## Cross-Platform Context

The MI300X uses only ~17.5% of its 192 GB VRAM -- dramatically over-provisioned for this workload. It achieves the best absolute results not through model size but through step throughput.

| Dataset | MI300X | A100 40GB | M5 Max |
|---------|--------|-----------|--------|
| ClimbMix | **1.036** | 1.142 | 1.296 |
| FineWeb-Edu | **1.015** | 1.194 | 1.342 |
| Cosmopedia-v2 | 1.015 | **0.697** | 0.961 |

See [Cross-Platform Overview](Cross-Platform-Overview) for the full comparison.

## Source Repository

- [elementalcollision/autoresearch-rocm](https://github.com/elementalcollision/autoresearch-rocm)

## Querying the Data

```python
from datasets import load_dataset
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
rocm = df[df["platform"] == "amd_rocm"]
```

See [Data Access](Data-Access) for more query examples.
