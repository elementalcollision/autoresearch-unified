---
license: cc-by-4.0
task_categories:
  - tabular-regression
  - tabular-classification
tags:
  - hyperparameter-optimization
  - autonomous-research
  - LLM-agent
  - GPU-benchmarks
  - cross-platform
  - language-model-training
pretty_name: Autoresearch Cross-Platform Experiments
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/experiments.parquet
  - config_name: hardware
    data_files:
      - split: train
        path: data/hardware.parquet
---

# Autoresearch Cross-Platform Experiments

## Dataset Description

This dataset contains **2,637 hyperparameter optimization experiments** from an autonomous LLM-driven ML research project. An LLM agent (Claude Sonnet) autonomously proposes hyperparameter modifications, trains a small language model for 5 minutes, evaluates validation bits-per-byte (val_bpb), and iterates.

Experiments span **3 hardware platforms**, **5 GPU models**, and **7 text datasets**, making this a unique resource for studying:
- Cross-platform hyperparameter transfer
- Hardware-adaptive optimization strategies
- LLM agent reasoning in automated ML research
- GPU cost-efficiency for language model training

## Quick Start

```python
from datasets import load_dataset

# Load all experiments
ds = load_dataset("elementalcollision/autoresearch-experiments")

# Load hardware reference table
hw = load_dataset("elementalcollision/autoresearch-experiments", "hardware")

# Filter to a specific platform
import pandas as pd
df = ds["train"].to_pandas()
cuda_results = df[df["platform"] == "nvidia_cuda"]
```

## Dataset Structure

### Experiments Table

| Column | Type | Description |
|--------|------|-------------|
| `experiment_id` | string | Globally unique: `{platform}_{gpu}_{dataset}_{run_id}_{exp}` |
| `platform` | string | `apple_metal`, `nvidia_cuda`, or `amd_rocm` |
| `gpu_name` | string | GPU model (M5 Max, RTX 4000 Ada, A100 40GB, RTX Pro 6000 Blackwell, MI300X) |
| `gpu_provider` | string | Cloud provider: `local`, `digitalocean`, `vultr`, `runpod` |
| `dataset` | string | Training dataset (climbmix, cosmopedia-v2, fineweb, fineweb-edu, fineweb-edu-high, github-code-python, slimpajama) |
| `agent_model` | string | LLM agent version: `sonnet-4.0` or `sonnet-4.6` |
| `run_id` | string | Experiment run identifier within a platform/GPU combination |
| `exp` | string | Experiment number (exp0 = baseline) |
| `description` | string | Agent's description of the hyperparameter change |
| `val_bpb` | float64 | **Primary metric**: validation bits-per-byte (lower = better; 0.0 = crash) |
| `peak_mem_gb` | float32 | Peak GPU memory usage (GB) |
| `tok_sec` | float64 | Training throughput (tokens/second) |
| `mfu` | float32 | Model FLOPs Utilization (%) |
| `steps` | float64 | Training steps completed in 5-minute budget |
| `status` | string | Outcome: `baseline`, `keep` (improved), `discard` (worse), `crash` |
| `notes` | string | Agent's reasoning and analysis |

### Hardware Reference Table

| Column | Type | Description |
|--------|------|-------------|
| `gpu_name` | string | GPU model name (primary key) |
| `platform` | string | Hardware platform |
| `architecture` | string | GPU architecture (Ada Lovelace, CDNA 3, etc.) |
| `vram_gb` | int | GPU memory (GB) |
| `bf16_tflops` | float | bf16 compute performance (TFLOPS) |
| `memory_bandwidth_gbps` | float | Memory bandwidth (GB/s) |
| `tdp_watts` | int | Thermal Design Power (W) |
| `cost_per_hour` | float | Cloud cost (USD/hr; $0 for local) |

## Dataset Statistics

| Dimension | Count |
|-----------|-------|
| Total experiments | 2,637 |
| Platforms | 3 (Apple Metal, NVIDIA CUDA, AMD ROCm) |
| GPU models | 5 |
| Datasets | 7 |
| NVIDIA CUDA experiments | 1,602 |
| Apple Metal experiments | 713 |
| AMD ROCm experiments | 322 |

## Understanding val_bpb

**Validation bits-per-byte (val_bpb)** is the primary metric. It measures how well the trained language model compresses held-out text:
- **Lower is better** — fewer bits needed per byte of text
- **0.0 means crash** — the training run failed (out-of-memory, NaN loss, timeout)
- **Typical range**: 0.7–1.6 depending on dataset complexity
- **exp0 is always the baseline** — subsequent experiments attempt to improve upon it

## Methodology

Each experiment follows this protocol:
1. The LLM agent reviews prior experiment results and proposes a hyperparameter modification
2. A small GPT-2-scale language model is trained for exactly 5 minutes
3. val_bpb is measured on a held-out validation set
4. The result is classified as `keep` (better than best so far), `discard` (worse), or `crash`
5. The agent uses this feedback to inform the next proposal

This is based on [Karpathy's autoresearch framework](https://github.com/karpathy/autoresearch), extended to support multiple hardware platforms and datasets.

## Source Repositories

| Platform | Repository | Wiki |
|----------|-----------|------|
| Apple Metal (MLX/MPS) | [autoresearch](https://github.com/elementalcollision/autoresearch) | [Wiki](https://github.com/elementalcollision/autoresearch/wiki) |
| NVIDIA CUDA | [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) | [Wiki](https://github.com/elementalcollision/autoresearch-cuda/wiki) |
| AMD ROCm | [autoresearch-rocm](https://github.com/elementalcollision/autoresearch-rocm) | [Wiki](https://github.com/elementalcollision/autoresearch-rocm/wiki) |
| Intel Gaudi | [autoresearch-gaudi](https://github.com/elementalcollision/autoresearch-gaudi) | [Wiki](https://github.com/elementalcollision/autoresearch-gaudi/wiki) |
| **Unified** | [autoresearch-unified](https://github.com/elementalcollision/autoresearch-unified) | [Wiki](https://github.com/elementalcollision/autoresearch-unified/wiki) |

## Croissant Compliance

This dataset conforms to the [MLCommons Croissant](https://mlcommons.org/croissant/) metadata standard (v1.0). The `croissant.json` file provides machine-readable dataset descriptions compatible with Google Dataset Search, HuggingFace, Kaggle, and other Croissant-aware platforms.

## Key Findings

1. **Architecture convergence**: 3 of 5 datasets on Apple Silicon converge to identical hyperparameters (AR=32)
2. **VRAM drives performance**: When constrained to the same model config, RTX 4000 and A100 achieve identical val_bpb — the A100's advantage comes from fitting larger models
3. **MI300X depth-steps tradeoff**: Reducing depth from 12→10 yielded 50% more training steps and better val_bpb
4. **Agent generation matters**: Sonnet 4.6 found 8 keeps vs 1 for Sonnet 4.0, with 20x greater improvement
5. **Cost-efficiency is non-linear**: RTX 4000 delivers 1.50 bpb/$ vs A100's 0.95 bpb/$

## License

This dataset is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

```bibtex
@dataset{autoresearch_experiments_2026,
  title={Autoresearch Cross-Platform Experiments},
  author={elementalcollision},
  year={2026},
  url={https://huggingface.co/datasets/elementalcollision/autoresearch-experiments},
  license={CC-BY-4.0}
}
```
