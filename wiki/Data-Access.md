# Data Access Guide

## HuggingFace Dataset

All experiment data is available as a Croissant-compliant dataset on HuggingFace:

**[davegraham/autoresearch-experiments](https://huggingface.co/datasets/davegraham/autoresearch-experiments)**

### Loading in Python

```python
from datasets import load_dataset

# Load all experiments
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()

# Load hardware reference table
hw = load_dataset("davegraham/autoresearch-experiments", "hardware")
hw_df = hw["train"].to_pandas()
```

### Common Queries

```python
import pandas as pd

# Best result per dataset per GPU
valid = df[df["val_bpb"] > 0]  # Exclude crashes
best = valid.groupby(["dataset", "gpu_name"])["val_bpb"].min().unstack()
print(best)

# Compare Sonnet 4.0 vs 4.6 on ClimbMix
climbmix = df[(df["dataset"] == "climbmix") & (df["platform"] == "apple_metal")]
for model, group in climbmix.groupby("agent_model"):
    keeps = group[group["status"] == "keep"]
    print(f"{model}: {len(keeps)} keeps, best={keeps['val_bpb'].min():.4f}")

# Cost-efficiency analysis
merged = valid.merge(hw_df[["gpu_name", "cost_per_hour"]], on="gpu_name")
merged["cost_per_bpb"] = merged["cost_per_hour"] / merged["val_bpb"]
```

### Direct Parquet Access

```python
import pandas as pd

# Without the datasets library
df = pd.read_parquet("hf://datasets/davegraham/autoresearch-experiments/data/experiments.parquet")
```

## Croissant Metadata

This dataset conforms to [MLCommons Croissant v1.0](https://mlcommons.org/croissant/). The metadata is available at:

```
https://huggingface.co/api/datasets/davegraham/autoresearch-experiments/croissant
```

### Validation

```bash
pip install mlcroissant
mlcroissant validate --jsonld croissant.json
```

### Loading via Croissant

```python
import mlcroissant as mlc

dataset = mlc.Dataset(jsonld="https://huggingface.co/api/datasets/davegraham/autoresearch-experiments/croissant")
records = dataset.records("experiments")
for record in records:
    print(record)
```

## Understanding the Data

### Key Fields

- **val_bpb**: Validation bits-per-byte (lower = better). This is the primary metric. A value of 0.0 indicates a crash.
- **status**: `baseline` (exp0, reference point), `keep` (improved over best), `discard` (worse), `crash` (training failed)
- **description**: The LLM agent's description of what hyperparameter change it tried
- **notes**: The agent's analysis of the result

### Experiment Protocol

1. exp0 is always the baseline (no modifications)
2. The agent proposes one hyperparameter change per experiment
3. Each training run is exactly 5 minutes
4. If val_bpb improves over the current best, status = `keep`

## Regenerating from Source

To regenerate the dataset from the raw `results.tsv` files:

```bash
cd autoresearch-unified
python -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow
python scripts/consolidate_results.py
```

This walks all 4 source repos and produces `data/experiments.parquet` and `data/hardware.parquet`.
