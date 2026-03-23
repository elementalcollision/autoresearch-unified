# Dataset: fineweb

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| A100 40GB | nvidia_cuda | vultr | 1.2314 | 98 | 9.2% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.2375 | 97 | 8.2% | 0.0% |
| RTX 4000 Ada | nvidia_cuda | digitalocean | 1.3285 | 2 | 0.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "fineweb") & (df["val_bpb"] > 0)]
```
