# Dataset: slimpajama

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| MI300X | amd_rocm | runpod | 1.0148 | 22 | 9.1% | 0.0% |
| A100 40GB | nvidia_cuda | vultr | 1.2450 | 96 | 8.3% | 0.0% |
| RTX 4000 Ada | nvidia_cuda | digitalocean | 1.3267 | 157 | 7.0% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.4579 | 50 | 6.0% | 0.0% |
| M5 Max | apple_metal | local | 1.5259 | 101 | 3.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "slimpajama") & (df["val_bpb"] > 0)]
```
