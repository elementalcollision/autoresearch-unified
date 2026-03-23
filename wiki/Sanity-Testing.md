# Sanity Testing: Unified Codebase

Initial validation results after merging all four platform-specific repositories into `autoresearch-unified`.

## Test Environment

| Property | Value |
|----------|-------|
| **Date** | 2026-03-23 |
| **Commit** | `5f77361` (Fix PYTHONPATH for training scripts relocated to platforms/) |
| **Hardware** | Apple M5 Max (40 GPU cores, 64 GB unified memory) |
| **Backend** | MLX (auto-detected) |
| **Dataset** | ClimbMix (default profile, cached at `~/.cache/autoresearch/`) |
| **Python** | 3.12.12 |
| **Key packages** | mlx 0.31.1, anthropic 0.86.0, textual 8.1.1, torch 2.10.0 |
| **LLM** | Claude Sonnet (claude-sonnet-4-20250514) via macOS Keychain |

## Integration Test Results

Ran 2-experiment headless suite (`--max 2 --tag unified-test3`):

| Exp | Description | val_bpb | Peak Mem | tok/s | MFU | Steps | Result |
|-----|-------------|---------|----------|-------|-----|-------|--------|
| exp0 | Baseline (no modifications) | **1.3669** | 13.7 GB | 50,898 | 12.0% | 468 | **KEEP** |
| exp1 | Increase DEVICE_BATCH_SIZE 1 to 2 | 1.3898 | 13.5 GB | 43,926 | 10.3% | 404 | DISCARD |

**Best val_bpb**: 1.3669 (baseline)

## What Was Validated

### Module Import Tests (all passed)

| Module | Status | Notes |
|--------|--------|-------|
| `backends.registry` | OK | All 6 platforms registered, all training scripts exist on disk |
| `backends.detect_backend()` | OK | Auto-detected MLX; helpful error when no GPU libs available |
| `backends.get_peak_flops()` | OK | 3.40e+13 FLOPS for M5 Max |
| `backends.suggest_hyperparameters()` | OK | depth=8, batch_size=16 for 64GB max tier |
| `tui.resilience` | OK | `atomic_write`, `atomic_append` work; `EXPECTED_FIELDS=10` |
| `tui.results` | OK | 10-column header with `gpu_name`; `ExperimentResult` dataclass |
| `tui.parser` | OK | `OutputParser.parse_line()` handles step + final output |
| `tui.experiments` | OK | Clean import |
| `tui.hardware` | OK | Detected Apple M5 Max via `sysctl` |
| `tui.credentials` | OK | Resolved API key via macOS Keychain |
| `tui.git_manager` | OK | `head_commit_message()`, `reset_working_tree()` present |
| `scripts.migrate_9col_to_10col` | OK | 9-col and 10-col headers verified |

### End-to-End Flow (passed)

The full autonomous experiment loop executed successfully:

1. **Backend detection** -- Auto-detected MLX on Apple M5 Max
2. **API connection** -- Connected to Claude Sonnet via Keychain credentials
3. **Git branch creation** -- Created `autoresearch/unified-test3` branch
4. **Baseline training** -- 5 minutes of GPT training, 468 steps, 15.3M tokens
5. **Output parsing** -- Step metrics and final evaluation parsed correctly
6. **Results recording** -- 10-column TSV with `gpu_name=Apple M5 Max`
7. **LLM experiment proposal** -- Claude proposed increasing DEVICE_BATCH_SIZE
8. **HP block modification** -- Committed changes to training script on branch
9. **Experiment training** -- Second 5-minute training run, 404 steps
10. **Comparison logic** -- Correctly discarded exp1 (1.3898 > 1.3669 baseline)
11. **Clean shutdown** -- Graceful exit with summary stats

### Bug Found and Fixed

**Issue**: Training scripts relocated from repo root to `platforms/<name>/` could not resolve `from prepare import ...` because Python adds the script's directory (not the repo root) to `sys.path`.

**Fix**: Set `PYTHONPATH` to the repo root in:
- `tui/orchestrator.py` (subprocess environment for training)
- `train.py` (dispatcher environment before `os.execv`)

**Commit**: `5f77361`

## Platforms Pending Testing

| Platform | Hardware | Status | Notes |
|----------|----------|--------|-------|
| Apple MLX | M5 Max | **Tested** | Full end-to-end pass |
| NVIDIA CUDA | RTX 4000 Ada / A100 / RTX Pro 6000 | Pending | Awaiting remote instance |
| AMD ROCm | MI300X | Pending | Awaiting remote instance |
| Intel Gaudi | Gaudi 3 | Pending | Awaiting remote instance |

## How to Run

```bash
# Install with platform-appropriate deps
uv pip install -e ".[all-metal]"   # Apple Silicon
uv pip install -e ".[all-cuda]"    # NVIDIA
uv pip install -e ".[all-rocm]"    # AMD

# Quick 2-experiment sanity test
python -m tui.headless --tag sanity --max 2 --results results/sanity/results.tsv

# Or use the TUI
python -m tui.app
```
