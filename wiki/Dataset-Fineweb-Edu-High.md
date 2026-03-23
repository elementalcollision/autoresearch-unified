# Dataset: fineweb-edu-high

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.0985 | 96 | 28.1% | 0.0% |
| A100 40GB | nvidia_cuda | vultr | 1.1067 | 96 | 25.0% | 0.0% |
| RTX 4000 Ada | nvidia_cuda | digitalocean | 1.1819 | 98 | 6.1% | 0.0% |
| M5 Max | apple_metal | local | 1.3463 | 96 | 20.8% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "fineweb-edu-high") & (df["val_bpb"] > 0)]
```
