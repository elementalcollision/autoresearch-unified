# Platform: NVIDIA CUDA

NVIDIA CUDA platform spanning three GPU architectures (Ada Lovelace, Ampere, Blackwell) across three cloud providers. Uses `torch.compile` with CUDA graphs and FlashAttention-2 for autonomous LLM-driven hyperparameter optimization.

## Summary

| Stat | Value |
|------|-------|
| **Platform** | `nvidia_cuda` |
| **Total experiments** | 1,602 |
| **GPUs tested** | RTX 4000 Ada (20 GB), A100 40GB, RTX Pro 6000 Blackwell (96 GB) |
| **Datasets** | 7 (ClimbMix, Cosmopedia-v2, FineWeb, FineWeb-Edu, FineWeb-Edu-High, SlimPajama, GitHub-Code-Python) |
| **Cloud providers** | DigitalOcean, Vultr, RunPod |

## Software Stack

| Component | Details |
|-----------|---------|
| **Framework** | PyTorch 2.6.0+cu124 |
| **Compilation** | `torch.compile` with CUDA graphs |
| **Attention** | FlashAttention-2 via PyTorch SDPA |
| **Precision** | bf16 autocast + tensor cores |
| **Optimizer** | Muon+AdamW (compiled step functions) |

## GPU Comparison

| GPU | Architecture | VRAM | bf16 TFLOPS | Provider | Cost/hr | Experiments |
|-----|-------------|------|-------------|----------|---------|-------------|
| **RTX 4000 Ada** | Ada Lovelace | 20 GB GDDR6 | ~105 | DigitalOcean | $0.76 | ~508 |
| **A100 40GB** | Ampere | 40 GB HBM2e | ~156 | Vultr | $1.20 | ~604 |
| **RTX Pro 6000** | Blackwell | 96 GB GDDR7 | ~260 | RunPod | $1.69 | ~490 |

## Results by Dataset and GPU

| Dataset | RTX 4000 Ada | A100 40GB | RTX Pro 6000 | Winner | Page |
|---------|-------------|-----------|--------------|--------|------|
| **ClimbMix** | — | **1.142** | 1.447 | A100 | [Dataset: ClimbMix](Dataset-Climbmix) |
| **Cosmopedia-v2** | 0.772 | **0.697** | — | A100 | [Dataset: Cosmopedia-v2](Dataset-Cosmopedia-V2) |
| **FineWeb-Edu** | **1.178** | 1.194 | 1.443 | RTX 4000 | [Dataset: FineWeb-Edu](Dataset-Fineweb-Edu) |
| **FineWeb-Edu-High** | 1.182 | 1.107 | **1.099** | Pro 6000 | [Dataset: FineWeb-Edu-High](Dataset-Fineweb-Edu-High) |
| **FineWeb** | 1.329 | **1.231** | 1.238 | A100 | [Dataset: FineWeb](Dataset-Fineweb) |
| **SlimPajama** | 1.327 | **1.245** | 1.458 | A100 | [Dataset: SlimPajama](Dataset-Slimpajama) |
| **GitHub-Code-Python** | — | **0.549** | 0.560 | A100 | [Dataset: GitHub-Code-Python](Dataset-Github-Code-Python) |

## Key Finding: VRAM Is King

The single biggest performance differentiator is VRAM, not raw TFLOPS.

When both the RTX 4000 (20 GB) and A100 (40 GB) are constrained to the same model configuration (AR=32, ~10.6 GB), they converge to **identical val_bpb** (1.142 on ClimbMix). When the A100 can fit larger models, it wins decisively:

| Dataset | RTX 4000 (20 GB) | A100 (40 GB) | A100 Config | Delta |
|---------|-------------------|--------------|-------------|-------|
| Cosmopedia-v2 | 0.772 | **0.697** | depth=9, AR=52, 24.7 GB | 9.7% better |
| SlimPajama | 1.327 | **1.245** | larger config, 2x throughput | 6.2% better |

## Blackwell: Different Optimization Landscape

The RTX Pro 6000 Blackwell shows a distinctly different convergence pattern from Ada/Ampere GPUs:

| Signal | Ada/Ampere | Blackwell |
|--------|------------|-----------|
| **Primary lever** | Architecture (AR, window pattern) | Learning rates (MATRIX_LR, SCALAR_LR) |
| **Optimal MATRIX_LR** | 0.04-0.05 | 0.019-0.032 |
| **Crash rate** | 3-8% | 10-50% |
| **Baseline quality** | Good defaults | Defaults need tuning |

Blackwell wins only on **FineWeb-Edu-High** (1.099 vs 1.107 for A100) -- the highest-quality dataset. Clean data unlocks its 96 GB VRAM and high throughput.

## Architecture Convergence

### Window pattern SLL is universally optimal

Across all three GPUs and all completed datasets, the agent converges to the **SLL** (or LL) window pattern -- never the default SSSL. This is the single most consistent finding in the study.

### Hardware-adaptive tier defaults

| Tier | GPUs | Depth | Device Batch | Total Batch |
|------|------|-------|-------------|-------------|
| Professional | RTX 4000/6000 Ada, RTX Pro 6000 | 10 | 32 | 64K tokens |
| Datacenter | A100, H100 | 12 | 64 | 128K tokens |

## Cloud Provider Details

| Provider | GPU | Region | Cost/hr | Experiments | Est. Total Cost |
|----------|-----|--------|---------|-------------|-----------------|
| **DigitalOcean** | RTX 4000 Ada | — | $0.76 | ~508 | ~$50 |
| **Vultr** | A100 40GB | — | $1.20 | ~604 | ~$96 |
| **RunPod** | RTX Pro 6000 | — | $1.69 | ~490 | ~$72 |

## Cost-Effectiveness

| GPU | $/hr | $/experiment | Best for |
|-----|------|-------------|----------|
| RTX 4000 Ada | $0.76 | ~$0.10 | Budget runs, VRAM-constrained exploration |
| A100 40GB | $1.20 | ~$0.16 | Large models, most datasets won |
| RTX Pro 6000 | $1.69 | ~$0.18 | High-quality data, Blackwell R&D |

## Source Repository

- [elementalcollision/autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda)
- [CUDA Wiki](https://github.com/elementalcollision/autoresearch-cuda/wiki) (per-GPU and per-dataset breakdowns)

## Querying the Data

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
cuda = df[df["platform"] == "nvidia_cuda"]
```

See [Data Access](Data-Access) for more query examples and [Cross-Platform Overview](Cross-Platform-Overview) for comparisons with Apple Metal and AMD ROCm.
