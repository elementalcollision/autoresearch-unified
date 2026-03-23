# Dataset: cosmopedia-v2

## Cross-Platform Results

| GPU | Platform | Provider | Best val_bpb | Experiments | Keep Rate | Crash Rate |
|-----|----------|----------|-------------|-------------|-----------|------------|
| A100 40GB | nvidia_cuda | vultr | 0.6972 | 92 | 21.7% | 0.0% |
| RTX 4000 Ada | nvidia_cuda | digitalocean | 0.7722 | 154 | 5.8% | 0.0% |
| M5 Max | apple_metal | local | 0.9606 | 101 | 4.0% | 0.0% |
| MI300X | amd_rocm | runpod | 1.0148 | 99 | 0.0% | 0.0% |

## Loading This Dataset

```python
from datasets import load_dataset
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
subset = df[(df["dataset"] == "cosmopedia-v2") & (df["val_bpb"] > 0)]
```
