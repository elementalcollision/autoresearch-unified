# Platform: Intel Gaudi

Intel Gaudi 3 HPU platform -- pre-production, with training infrastructure deployed but no autonomous experiment runs completed yet. The port targets IBM Cloud Gaudi 3 instances with SynapseAI and a custom Habana PyTorch build.

## Summary

| Stat | Value |
|------|-------|
| **Platform** | `intel_gaudi` |
| **Total experiments** | 0 (pre-production) |
| **GPU** | Intel Gaudi 3 HPU |
| **Datasets** | 0 (pending deployment) |
| **Provider** | IBM Cloud (`gx3d-160x1792x8gaudi3`) |
| **Status** | Infrastructure deployed, awaiting experiment runs |

## Hardware Specifications

| Spec | Gaudi 3 |
|------|---------|
| **Accelerators per instance** | 8 |
| **Memory per device** | 128 GB HBM2e |
| **Memory bandwidth** | ~3.7 TB/s |
| **bf16 TFLOPS per device** | ~1,835 |
| **Tensor Processor Cores** | 64 per device |
| **Monitoring** | `hl-smi` |

## Software Stack

| Component | Details |
|-----------|---------|
| **Runtime** | SynapseAI 1.23.0 |
| **Container** | `vault.habana.ai/gaudi-docker/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest` |
| **PyTorch** | Custom Habana build (do NOT `pip install torch` -- HPU backend is compiled in) |
| **Compilation** | `torch.compile(backend="hpu_backend")` |
| **Attention** | FusedSDPA / SDPA |
| **Device** | `torch.device("hpu")` |
| **Deployment** | Docker Compose (build, verify, prepare, train, agent) |

## Default Hyperparameters

Scaled for Gaudi 3's compute and memory capacity:

| Parameter | Value |
|-----------|-------|
| **DEPTH** | 16 |
| **DEVICE_BATCH_SIZE** | 64 |
| **TOTAL_BATCH_SIZE** | 262,144 tokens |
| **ASPECT_RATIO** | 64 |
| **TIME_BUDGET** | 300s |

These defaults are significantly larger than other platforms (depth 16 vs 8-12, batch 256K vs 32-128K tokens), reflecting the Gaudi 3's 128 GB HBM2e and ~1,835 bf16 TFLOPS per device.

## Deployment Preparation

The infrastructure is fully deployed:

- Training script (`train_gaudi.py`) -- single HPU, GPT with ResFormer value embeddings, RoPE, Muon+AdamW
- Headless agent mode (`tui/headless.py`) for unattended experiment loops
- Multi-dataset suite (`run_suite.py`) for full 7-dataset sweeps
- Docker Compose workflow for IBM Cloud Gaudi 3 instances
- IBM Cloud setup script (`setup_ibm_cloud.sh`)
- HPU verification and benchmarking scripts

### Quick Start (on IBM Cloud)

```bash
bash setup_ibm_cloud.sh          # One-time instance setup
docker compose build              # Build container
docker compose run verify         # Verify HPU access
docker compose run prepare        # Download + tokenize data
docker compose run agent          # Run autonomous experiments
```

## Differences from Other Platforms

| Aspect | CUDA | Apple Silicon | ROCm | Gaudi 3 |
|--------|------|---------------|------|---------|
| Device | `cuda` | `mps` | `cuda` (ROCm) | `hpu` |
| Compile | `torch.compile()` | Eager / `mx.compile` | `torch.compile()` | `torch.compile(backend="hpu_backend")` |
| Attention | FlashAttention-2 | SDPA / MLX native | CK FlashAttention | FusedSDPA |
| Memory | 20-96 GB VRAM | 64 GB unified | 192 GB HBM3 | 128 GB HBM2e |

## Source Repository

- [elementalcollision/autoresearch-gaudi](https://github.com/elementalcollision/autoresearch-gaudi)

## Querying the Data

No experiment data is available yet. Once runs begin, data will appear under `platform == "intel_gaudi"`:

```python
from datasets import load_dataset
ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
gaudi = df[df["platform"] == "intel_gaudi"]
```

See [Data Access](Data-Access) for general query examples and [Cross-Platform Overview](Cross-Platform-Overview) for comparisons across active platforms.
