# Dataset: climbmix

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| MI300X | amd_rocm | runpod | 1.0359 | 97 | 16.5% | 0.0% |
| A100 40GB | nvidia_cuda | vultr | 1.1422 | 94 | 24.5% | 0.0% |
| M5 Max | apple_metal | local | 1.2959 | 204 | 3.9% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.4468 | 50 | 12.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "climbmix") & (df["val_bpb"] > 0)]
```
