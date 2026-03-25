# autoresearch-unified

Autonomous LLM-driven GPU pretraining research -- unified across NVIDIA, AMD, Intel, and Apple platforms.

> **[Take the Interactive Course](https://elementalcollision.github.io/autoresearch-unified/)** — Learn how this entire codebase works through animated diagrams, code walkthroughs, and interactive quizzes. No CS background needed.

## What Is This?

An autonomous experiment loop where **Claude proposes hyperparameter changes, trains a GPT-2-scale language model for 5 minutes, evaluates val_bpb (validation bits-per-byte), and decides to keep or discard** -- repeating across hundreds of experiments to find optimal configurations on each hardware platform.

This repository is the **unified codebase** that runs on any supported platform while exploiting each platform's unique capabilities. It consolidates four previously separate repositories:

| Platform | Training Backend | Hardware | Source Repo |
|----------|-----------------|----------|-------------|
| Apple Metal | MLX | M1 Max, M4 Pro, M5 Max | [autoresearch](https://github.com/elementalcollision/autoresearch) |
| NVIDIA CUDA | PyTorch + torch.compile | RTX 4000 Ada, A100 40GB, RTX Pro 6000 | [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda) |
| AMD ROCm | PyTorch + SDPA/CK | MI300X | [autoresearch-rocm](https://github.com/elementalcollision/autoresearch-rocm) |
| Intel Gaudi | PyTorch + hpu_backend | Gaudi 3 | [autoresearch-gaudi](https://github.com/elementalcollision/autoresearch-gaudi) |

## Quick Start

```bash
# Clone
git clone https://github.com/elementalcollision/autoresearch-unified.git
cd autoresearch-unified

# Install (pick your platform)
uv pip install -e ".[all-metal]"   # Apple Silicon (MLX + agent + analysis)
uv pip install -e ".[all-cuda]"    # NVIDIA CUDA
uv pip install -e ".[all-rocm]"    # AMD ROCm

# Prepare data (downloads ClimbMix, trains tokenizer)
python prepare.py

# Run the experiment loop (headless, 80 experiments)
python -m tui.headless --tag my-run --max 80

# Or use the interactive TUI
python -m tui.app

# Or run a multi-dataset overnight suite
python run_suite.py --max-per-dataset 80
```

The backend is auto-detected, or override with `AUTORESEARCH_BACKEND=cuda|rocm|rocm7|mlx|mps|hpu`.

## How It Works

```
Claude proposes HP changes --> git commit --> train 5 min --> evaluate val_bpb
     ^                                                              |
     |                          keep / discard <--------------------+
     +--------------------------------------------------------------+
```

1. **Claude** (via Anthropic API) analyzes past results and proposes a hyperparameter change
2. The change is **committed to a git branch** for reproducibility
3. The platform-specific **training script runs for 5 minutes** (time-budgeted)
4. **val_bpb** is evaluated on a pinned validation shard
5. If val_bpb improves over the best so far, the experiment is **kept**; otherwise **discarded**
6. Repeat up to N experiments

## Repository Structure

```
autoresearch-unified/
├── train.py                    # Universal dispatcher (auto-detects platform)
├── prepare.py                  # Data download + tokenizer training
├── run_suite.py                # Multi-dataset overnight suite runner
├── dashboard.py                # Results visualization
├── compare_datasets.py         # Cross-dataset comparison
├── compare_backends.py         # Cross-platform comparison
├── monitor.py                  # Live training monitor
│
├── tui/                        # Shared application layer
│   ├── app.py                  # Interactive Textual TUI
│   ├── orchestrator.py         # Core experiment loop
│   ├── headless.py             # Headless runner (no terminal needed)
│   ├── llm_backend.py          # Claude API integration
│   ├── results.py              # 10-column TSV with atomic writes
│   ├── resilience.py           # Heartbeat, signal handlers, PID lock
│   ├── credentials.py          # API key (Keychain on macOS, file on Linux)
│   ├── parser.py               # Training output parser
│   ├── hardware.py             # Hardware detection (all platforms)
│   ├── git_manager.py          # Git branch/commit management
│   ├── widgets.py              # TUI widgets
│   └── experiments.py          # Experiment loading from TSV
│
├── backends/                   # Platform detection + optimizers
│   ├── __init__.py             # detect_backend(), get_hardware_info()
│   ├── registry.py             # Platform metadata (scripts, FLOPS, names)
│   ├── muon_cuda.py            # MuonAdamW for NVIDIA
│   ├── muon_rocm.py            # MuonAdamW for AMD ROCm 6.x
│   ├── muon_rocm7.py           # MuonAdamW for AMD ROCm 7.x
│   ├── muon_gaudi.py           # MuonAdamW for Intel Gaudi
│   ├── muon_mlx.py             # MuonAdamW for Apple MLX
│   └── muon_mps.py             # MuonAdamW for Apple MPS
│
├── platforms/                  # Platform-specific training scripts
│   ├── metal/train_mlx.py      # Apple MLX training
│   ├── cuda/train_cuda.py      # NVIDIA CUDA training
│   ├── rocm/train_rocm.py      # AMD ROCm 6.x training
│   ├── rocm/train_rocm7.py     # AMD ROCm 7.x training
│   └── gaudi/                  # Intel Gaudi training + deployment
│       ├── train_gaudi.py
│       ├── Dockerfile
│       ├── docker-compose.yml
│       └── scripts/            # HPU verification, benchmarking, launch
│
├── data/                       # Consolidated Parquet (experiment results)
├── results/                    # Per-dataset result directories
├── scripts/                    # Data consolidation + HuggingFace upload
├── huggingface/                # HF dataset card + Croissant metadata
└── wiki/                       # Wiki page sources
```

## Platform Selection

Auto-detection priority: **HPU > ROCm > CUDA > MLX > MPS**

| Backend | Env Var | Requires |
|---------|---------|----------|
| NVIDIA CUDA | `AUTORESEARCH_BACKEND=cuda` | `torch` with CUDA |
| AMD ROCm 6.x | `AUTORESEARCH_BACKEND=rocm` | `torch` with ROCm |
| AMD ROCm 7.x | `AUTORESEARCH_BACKEND=rocm7` | `torch` with ROCm 7+ |
| Apple MLX | `AUTORESEARCH_BACKEND=mlx` | `mlx` (macOS only) |
| Apple MPS | `AUTORESEARCH_BACKEND=mps` | `torch` (macOS only) |
| Intel Gaudi | `AUTORESEARCH_BACKEND=hpu` | Habana Docker container |

## Adding a New Platform

1. Create `platforms/<name>/train_<name>.py` -- self-contained training script
2. Create `backends/muon_<name>.py` -- optimizer with platform-appropriate compile decorator
3. Add entry to `backends/registry.py` PLATFORMS dict
4. Add detection logic to `backends/__init__.py`
5. Add hardware query to `tui/hardware.py`

No changes needed to the orchestrator, results, resilience, TUI, or any other shared code.

## Experiment Data

**HuggingFace Dataset**: [davegraham/autoresearch-experiments](https://huggingface.co/datasets/davegraham/autoresearch-experiments)

```python
from datasets import load_dataset

ds = load_dataset("davegraham/autoresearch-experiments")
df = ds["train"].to_pandas()
```

- **2,637+ experiments** across 4 platforms, 5+ GPUs, 7 datasets
- Croissant-compliant (MLCommons v1.1), indexed on Google Dataset Search
- 10-column TSV: exp, description, val_bpb, peak_mem_gb, tok_sec, mfu, steps, status, notes, gpu_name

## Documentation

**[Interactive Course](https://elementalcollision.github.io/autoresearch-unified/)** — A visual, scroll-based walkthrough of the entire codebase with animated data flows, code-to-English translations, and quizzes. Great for onboarding or understanding the architecture.

See the [wiki](https://github.com/elementalcollision/autoresearch-unified/wiki) for:
- [Cross-Platform Overview](https://github.com/elementalcollision/autoresearch-unified/wiki/Cross-Platform-Overview) -- Key findings and normalized comparisons
- [Sanity Testing](https://github.com/elementalcollision/autoresearch-unified/wiki/Sanity-Testing) -- Integration test results for the unified codebase
- Platform-specific guides ([Metal](https://github.com/elementalcollision/autoresearch-unified/wiki/Platform-Apple-Metal), [CUDA](https://github.com/elementalcollision/autoresearch-unified/wiki/Platform-NVIDIA-CUDA), [ROCm](https://github.com/elementalcollision/autoresearch-unified/wiki/Platform-AMD-ROCm), [Gaudi](https://github.com/elementalcollision/autoresearch-unified/wiki/Platform-Intel-Gaudi))
- Per-dataset results ([ClimbMix](https://github.com/elementalcollision/autoresearch-unified/wiki/Dataset-Climbmix), [FineWeb-Edu](https://github.com/elementalcollision/autoresearch-unified/wiki/Dataset-FineWeb-Edu), [Cosmopedia-v2](https://github.com/elementalcollision/autoresearch-unified/wiki/Dataset-Cosmopedia-v2), and more)
- [Data Access Guide](https://github.com/elementalcollision/autoresearch-unified/wiki/Data-Access)

## License

Code: MIT | Data: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
