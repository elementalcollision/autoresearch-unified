# Cross-Platform Overview

## Best val_bpb by Dataset and GPU

| Dataset | MI300X | A100 40GB | RTX 4000 Ada | RTX Pro 6000 | M5 Max |
|---------|--------|-----------|--------------|--------------|--------|
| ClimbMix | **1.036** | 1.142 | — | 1.447 | 1.296 |
| Cosmopedia-v2 | 1.015 | **0.697** | 0.772 | — | 0.961 |
| FineWeb | — | **1.231** | 1.328 | 1.237 | — |
| FineWeb-Edu | **1.015** | 1.194 | 1.178 | 1.443 | 1.342 |
| FineWeb-Edu-High | — | 1.107 | 1.182 | **1.099** | 1.346 |
| SlimPajama | **1.015** | 1.245 | 1.327 | 1.458 | 1.526 |
| GitHub-Code-Python | — | **0.549** | — | 0.560 | — |

Bold = best result for that dataset.

## Key Findings

### 1. Architecture Convergence Across Datasets

Three of five datasets on Apple Silicon converge to identical hyperparameters (AR=32). Educational text (FineWeb-Edu variants) diverges, suggesting **data-dependent rather than path-dependent** optimization.

### 2. VRAM Is the Primary Performance Driver

When both RTX 4000 (20 GB) and A100 (40 GB) are constrained to the same model configuration, they produce identical val_bpb. The A100's advantage comes from fitting larger models (depth 8-9, AR 52-56), not from raw compute.

### 3. The MI300X Depth-Steps Tradeoff

Reducing depth from 12 to 10 on MI300X yielded 50% more training steps in the 5-minute budget (880→911) and dropped peak memory from 47.6→33.6 GB. **More steps beat more parameters when training time is fixed.**

### 4. LLM Agent Generation Matters

Sonnet 4.6 found 8 keeps vs 1 for Sonnet 4.0 on ClimbMix, achieved 20x greater improvement, and demonstrated compositional reasoning absent in 4.0. The optimizer's intelligence is itself a variable.

### 5. Cost-Efficiency Is Not Linear

| GPU | Cost/hr | Best ClimbMix val_bpb | bpb/$ |
|-----|---------|----------------------|-------|
| M5 Max | $0 (local) | 1.296 | ∞ |
| RTX 4000 Ada | $0.76 | — | — |
| A100 40GB | $1.20 | 1.142 | 0.95 |
| RTX Pro 6000 | $1.69 | 1.447 | 0.86 |
| MI300X | $1.99 | 1.036 | 0.52 |

The MI300X achieves the best absolute result but uses only 17.5% of its 192 GB VRAM — dramatically over-provisioned for this workload.

## Hardware Specifications

| Spec | M5 Max | RTX 4000 Ada | A100 40GB | RTX Pro 6000 | MI300X |
|------|--------|--------------|-----------|--------------|--------|
| **Architecture** | Apple GPU | Ada Lovelace | Ampere | Blackwell | CDNA 3 |
| **Memory** | 64 GB unified | 20 GB GDDR6 | 40 GB HBM2e | 96 GB GDDR7 | 192 GB HBM3 |
| **bf16 TFLOPS** | ~14 | ~105 | ~156 | ~260 | 1,307 |
| **Memory BW** | ~400 GB/s | 280 GB/s | 1,555 GB/s | 1,100 GB/s | 5,300 GB/s |
| **TDP** | ~92W | 130W | 300W | 250W | 750W |
| **Cost/hr** | $0 (local) | $0.76 | $1.20 | $1.69 | $1.99 |

## Software Stacks

| Component | Apple Metal | NVIDIA CUDA | AMD ROCm |
|-----------|-----------|-------------|----------|
| **Framework** | MLX / PyTorch | PyTorch 2.6.0+cu124 | PyTorch 2.9.1+rocm6.3 |
| **Compilation** | Eager / mx.compile | torch.compile + CUDA graphs | torch.compile + AMD Triton |
| **Attention** | SDPA / MLX native | FlashAttention-2 via SDPA | CK-based FlashAttention |
| **Precision** | bf16 manual casting | bf16 autocast + tensor cores | bf16 autocast |
