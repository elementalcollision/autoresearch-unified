# Dataset: github-code-python

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| A100 40GB | nvidia_cuda | vultr | 0.5492 | 30 | 26.7% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 0.5601 | 11 | 9.1% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "github-code-python") & (df["val_bpb"] > 0)]
```
