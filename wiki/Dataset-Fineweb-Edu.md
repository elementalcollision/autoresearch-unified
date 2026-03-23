# Dataset: fineweb-edu

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| MI300X | amd_rocm | runpod | 1.0148 | 99 | 14.1% | 0.0% |
| RTX 4000 Ada | nvidia_cuda | digitalocean | 1.1779 | 97 | 7.2% | 0.0% |
| A100 40GB | nvidia_cuda | vultr | 1.1945 | 98 | 7.1% | 0.0% |
| M5 Max | apple_metal | local | 1.3416 | 185 | 18.9% | 0.0% |
| RTX Pro 6000 Blackwell | nvidia_cuda | runpod | 1.4429 | 50 | 2.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("elementalcollision/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "fineweb-edu") & (df["val_bpb"] > 0)]
```
