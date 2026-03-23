# Platform: Apple Metal

Apple Silicon platform using a dual-backend architecture (MLX + PyTorch MPS) for autonomous LLM-driven hyperparameter optimization on local Mac hardware. All experiments run on unified CPU/GPU memory with no cloud dependency.

## Summary

| Stat | Value |
|------|-------|
| **Platform** | `apple_metal` |
| **Total experiments** | 713 |
| **GPUs tested** | M1 Max (64 GB), M4 Pro (24 GB), M5 Max (64 GB) |
| **Datasets** | 5 (ClimbMix, Cosmopedia-v2, FineWeb-Edu, FineWeb-Edu-High, SlimPajama) |
| **Primary GPU** | Apple M5 Max (64 GB unified) |
| **Cost** | $0 (local hardware) |

## Software Stack

| Component | Details |
|-----------|---------|
| **Primary backend** | MLX (`mx.compile`, native attention) |
| **Secondary backend** | PyTorch MPS (SDPA, eager mode) |
| **Precision** | bf16 manual casting (MPS) / native bf16 (MLX) |
| **Optimizer** | Muon+AdamW (novel MLX port of Newton-Schulz orthogonalization) |
| **Compilation** | `mx.compile` (MLX) / eager mode (MPS -- `torch.compile` not supported) |
| **Backend selection** | Auto-detected (prefers MLX); override via `AUTORESEARCH_BACKEND=mlx|mps` |

## Results by Dataset

| Dataset | GPU | Best val_bpb | Experiments | Keep Rate | Page |
|---------|-----|-------------|-------------|-----------|------|
| ClimbMix | M5 Max | 1.296 | 204 | 3.9% | [Dataset: ClimbMix](Dataset-Climbmix) |
| FineWeb-Edu | M5 Max | 1.342 | 185 | 18.9% | [Dataset: FineWeb-Edu](Dataset-Fineweb-Edu) |
| Cosmopedia-v2 | M5 Max | 0.961 | 101 | 4.0% | [Dataset: Cosmopedia-v2](Dataset-Cosmopedia-V2) |
| FineWeb-Edu-High | M5 Max | 1.346 | 96 | 20.8% | [Dataset: FineWeb-Edu-High](Dataset-Fineweb-Edu-High) |
| SlimPajama | M5 Max | 1.526 | 101 | 3.0% | [Dataset: SlimPajama](Dataset-Slimpajama) |

## Hardware Comparison (Apple Silicon Chips)

| Chip | Memory | Best val_bpb | Optimal Batch | Peak VRAM | Steps |
|------|--------|-------------|---------------|-----------|-------|
| **M5 Max** | 64 GB | **1.320** | 32K total, 16 device | 26.1 GB | 312 |
| M4 Pro | 24 GB | 1.429 | 8K total, 4 device | 4.5 GB | 751 |
| M1 Max | 64 GB | 1.621 | 16K total, 8 device | 11.3 GB | ~210 |

Key insight: maximizing optimizer steps within the fixed 5-minute time budget is the dominant factor. Each generation finds its own optimal batch size, balancing gradient quality against step throughput.

## Architecture Convergence

All five datasets on M5 Max converge to **AR=32** (hardware-optimal for 64 GB unified memory). Three of five converge to the exact same hyperparameters. Both FineWeb-Edu variants diverge, confirming that educational text has different optimization needs that are **data-dependent, not path-dependent**.

| Parameter | Default | ClimbMix / Cosmopedia / SlimPajama | FineWeb-Edu variants |
|-----------|---------|-------------------------------------|---------------------|
| **ASPECT_RATIO** | 64 | 32 | 32 |
| **Hyperparameters** | (default) | Identical cluster | Divergent cluster |

## Sonnet 4.6 vs 4.0 Comparison

Agent model quality is itself a variable. On ClimbMix:

- **Sonnet 4.6**: 8 keeps, 20x greater improvement, compositional reasoning (multi-parameter edits)
- **Sonnet 4.0**: 1 keep, minimal improvement, single-parameter edits only

This finding motivated the use of Sonnet 4.6 as the default agent across all platforms.

## Cross-Platform Context

Apple Metal results are generally higher (worse) val_bpb than CUDA or ROCm due to lower raw compute (~14 bf16 TFLOPS for M5 Max vs ~156 for A100). However, the $0 cost makes it ideal for rapid prototyping and validating experiment methodology before deploying to cloud GPUs.

| Dataset | M5 Max | A100 40GB | MI300X |
|---------|--------|-----------|--------|
| ClimbMix | 1.296 | 1.142 | **1.036** |
| FineWeb-Edu | 1.342 | 1.194 | **1.015** |
| Cosmopedia-v2 | 0.961 | **0.697** | 1.015 |

See [Cross-Platform Overview](Cross-Platform-Overview) for the full comparison.

## Source Repository

- [elementalcollision/autoresearch](https://github.com/elementalcollision/autoresearch) (Apple Silicon fork)
- [Apple Silicon Wiki](https://github.com/elementalcollision/autoresearch/wiki) (per-chip and per-dataset breakdowns)

## Querying the Data

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
metal = df[df["platform"] == "apple_metal"]
```

See [Data Access](Data-Access) for more query examples.
