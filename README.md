# autoresearch-unified

Consolidated data, documentation, and tooling for the Autoresearch cross-platform hyperparameter optimization project.

## What Is This?

This repo is the **single source of truth** for all Autoresearch experiment data and documentation, unifying results from 4 platform-specific forks:

| Platform | Source Repo | Hardware |
|----------|-----------|----------|
| Apple Metal | [autoresearch](https://github.com/elementalcollision/autoresearch) | M1 Max, M4 Pro, M5 Max |
| NVIDIA CUDA | [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) | RTX 4000 Ada, A100 40GB, RTX Pro 6000 Blackwell |
| AMD ROCm | [autoresearch-rocm](https://github.com/elementalcollision/autoresearch-rocm) | MI300X |
| Intel Gaudi | [autoresearch-gaudi](https://github.com/elementalcollision/autoresearch-gaudi) | Gaudi 3 (pending) |

## Data Access

**HuggingFace Dataset**: [davegraham/autoresearch-experiments](https://huggingface.co/datasets/davegraham/autoresearch-experiments)

```python
from datasets import load_dataset

ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
```

The dataset is [Croissant-compliant](https://mlcommons.org/croissant/) (MLCommons v1.0) and indexed on Google Dataset Search.

## Dataset Summary

- **2,637 experiments** across 3 platforms, 5 GPUs, 7 datasets
- Primary metric: **val_bpb** (validation bits-per-byte, lower is better)
- Each experiment = 5 minutes of training a GPT-2-scale language model
- Hyperparameters proposed autonomously by Claude Sonnet

## Repository Structure

```
autoresearch-unified/
├── scripts/
│   ├── consolidate_results.py   # Merge all results.tsv → Parquet
│   └── upload_to_hf.py          # Push to HuggingFace
├── data/
│   ├── experiments.parquet       # Unified experiment results
│   └── hardware.parquet          # GPU specs reference table
├── huggingface/
│   ├── README.md                 # HuggingFace Dataset Card
│   └── croissant.json            # MLCommons Croissant metadata
├── wiki/                         # Consolidated wiki page sources
└── README.md                     # This file
```

## Regenerating the Dataset

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow huggingface_hub

# Consolidate all results.tsv files into Parquet
python scripts/consolidate_results.py

# Upload to HuggingFace (requires `huggingface-cli login`)
python scripts/upload_to_hf.py
```

## Documentation

See the [wiki](https://github.com/elementalcollision/autoresearch-unified/wiki) for:
- Cross-platform experiment analysis
- Per-dataset results across all hardware
- Platform-specific configuration guides
- Tool documentation (TUI dashboard, remote orchestration)

## License

Code: MIT | Data: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
