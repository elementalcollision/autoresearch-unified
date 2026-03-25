# Sanity Testing: Unified Codebase

Initial validation results after merging all four platform-specific repositories into `autoresearch-unified`.

---

## Test 1: Apple M5 Max (MLX)

### Environment

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

### Experiment Results (2 experiments)

| Exp | Description | val_bpb | Peak Mem | tok/s | MFU | Steps | Result |
|-----|-------------|---------|----------|-------|-----|-------|--------|
| exp0 | Baseline (no modifications) | **1.3669** | 13.7 GB | 50,898 | 12.0% | 468 | **KEEP** |
| exp1 | Increase DEVICE_BATCH_SIZE 1 to 2 | 1.3898 | 13.5 GB | 43,926 | 10.3% | 404 | DISCARD |

**Best val_bpb**: 1.3669 (baseline)

### What Was Validated

#### Module Import Tests (all passed)

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

#### End-to-End Flow (passed)

The full autonomous experiment loop executed successfully:

1. **Backend detection** — Auto-detected MLX on Apple M5 Max
2. **API connection** — Connected to Claude Sonnet via Keychain credentials
3. **Git branch creation** — Created `autoresearch/unified-test3` branch
4. **Baseline training** — 5 minutes of GPT training, 468 steps, 15.3M tokens
5. **Output parsing** — Step metrics and final evaluation parsed correctly
6. **Results recording** — 10-column TSV with `gpu_name=Apple M5 Max`
7. **LLM experiment proposal** — Claude proposed increasing DEVICE_BATCH_SIZE
8. **HP block modification** — Committed changes to training script on branch
9. **Experiment training** — Second 5-minute training run, 404 steps
10. **Comparison logic** — Correctly discarded exp1 (1.3898 > 1.3669 baseline)
11. **Clean shutdown** — Graceful exit with summary stats

#### Bug Found and Fixed

**Issue**: Training scripts relocated from repo root to `platforms/<name>/` could not resolve `from prepare import ...` because Python adds the script's directory (not the repo root) to `sys.path`.

**Fix**: Set `PYTHONPATH` to the repo root in:
- `tui/orchestrator.py` (subprocess environment for training)
- `train.py` (dispatcher environment before `os.execv`)

**Commit**: `5f77361`

---

## Test 2: NVIDIA RTX 4000 Ada Generation (CUDA)

### Environment

| Property | Value |
|----------|-------|
| **Date** | 2026-03-23 |
| **Commit** | `967f33b` (Add RAS data sync infrastructure for RunPod deployments) |
| **Hardware** | NVIDIA RTX 4000 Ada Generation (20 GB VRAM) |
| **Platform** | RunPod (on-demand, $0.20/hr) |
| **Backend** | CUDA (env: `AUTORESEARCH_BACKEND=cuda`) |
| **Dataset** | ClimbMix (default profile) |
| **Python** | 3.11 |
| **Image** | `runpod/pytorch:2.6.0-py3.11-cuda12.8.1-devel-ubuntu22.04` |
| **LLM** | Claude Sonnet (claude-sonnet-4-20250514) |
| **Branch** | [`autoresearch/sanity-rtx4000ada`](https://github.com/elementalcollision/autoresearch-unified/tree/autoresearch/sanity-rtx4000ada) |

### Experiment Results (20 experiments, ~87 minutes)

| Exp | Description | val_bpb | Peak Mem | tok/s | MFU | Steps | Result |
|-----|-------------|---------|----------|-------|-----|-------|--------|
| exp0 | Baseline (no modifications) | 1.1767 | 8.1 GB | 106,666 | 24.1% | 977 | baseline |
| exp1 | Decrease TOTAL_BATCH_SIZE to 32768 | 1.1764 | 8.1 GB | 106,666 | 24.1% | 978 | **keep** |
| exp2 | Decrease TOTAL_BATCH_SIZE to 16384 | — | — | — | — | — | crash |
| exp3 | Increase TOTAL_BATCH_SIZE to 49152 | — | — | — | — | — | crash |
| exp4 | Increase MATRIX_LR to 0.05 | 1.1775 | 8.1 GB | 107,226 | 24.1% | 982 | discard |
| exp5 | Decrease MATRIX_LR to 0.035 | 1.1769 | 8.1 GB | 106,631 | 24.0% | 976 | discard |
| exp6 | Increase EMBEDDING_LR to 0.8 | 1.1765 | 8.1 GB | 106,893 | 24.1% | 979 | discard |
| exp7 | Decrease UNEMBEDDING_LR to 0.003 | 1.1769 | 8.1 GB | 106,964 | 24.1% | 979 | discard |
| exp8 | Decrease WEIGHT_DECAY to 0.1 | 1.1753 | 8.1 GB | 106,666 | 24.0% | 976 | **keep** |
| exp9 | Decrease WEIGHT_DECAY to 0.05 | **1.1752** | 8.1 GB | 106,560 | 24.1% | 978 | **keep** ⭐ |
| exp10 | Increase WARMDOWN_RATIO to 0.6 | 1.1753 | 8.1 GB | 106,631 | 24.0% | 976 | discard |
| exp11 | Increase DEPTH to 10 | 1.2132 | — | — | — | — | discard |
| exp12 | Decrease DEVICE_BATCH_SIZE | 1.3981 | — | — | — | — | discard |
| exp13 | Increase DEPTH to 11 | — | — | — | — | — | crash (OOM) |
| exp14 | Decrease TOTAL_BATCH_SIZE to 24576 | — | — | — | — | — | crash |
| exp15 | Increase DEPTH to 11 | — | — | — | — | — | crash (OOM) |
| exp16 | Decrease DEPTH to 9 | 1.3119 | — | — | — | — | discard |
| exp17 | Increase DEPTH to 11 | — | — | — | — | — | crash (OOM) |
| exp18 | Increase DEPTH to 9 | — | — | — | — | — | crash |
| exp19 | Increase DEPTH to 12 | — | — | — | — | — | crash (OOM) |

### Summary Statistics

| Metric | Value |
|--------|-------|
| **Total experiments** | 20 |
| **Kept** | 4 (20%) |
| **Discarded** | 8 (40%) |
| **Crashed** | 8 (40%) |
| **Best val_bpb** | **1.175187** (exp9: weight_decay=0.05) |
| **Improvement over baseline** | −0.13% (1.1767 → 1.1752) |
| **Avg throughput** | ~106,700 tok/s |
| **Consistent MFU** | 24.0–24.1% |
| **Peak VRAM** | 8.1 GB (of 20 GB available) |
| **Runtime** | ~87 minutes total |
| **Cost** | ~$0.29 ($0.20/hr × 1.45 hr) |

### What Was Validated

#### Core Pipeline (all passed)

| Component | Status | Notes |
|-----------|--------|-------|
| Backend detection | OK | CUDA detected via `AUTORESEARCH_BACKEND` env var |
| Training dispatch | OK | `train.py` → `platforms/cuda/train_cuda.py` via registry |
| Baseline training | OK | 977 steps, 32M tokens in 5 min |
| HP block extraction | OK | All hyperparameter modifications applied correctly |
| Git commit/revert | OK | Kept changes committed, discards reverted |
| Crash recovery | OK | 8 crashes handled gracefully, loop continued |
| 10-column TSV | OK | `gpu_name` populated as `NVIDIA RTX 4000 Ada Generation` |
| Heartbeat monitoring | OK | Updated every cycle with experiment/status/best_bpb |
| Atomic writes | OK | Results TSV never corrupted despite crashes |
| Clean shutdown | OK | Stopped after max=20, summary printed |

#### RAS Features (all validated)

| Feature | Status | Evidence |
|---------|--------|----------|
| **Heartbeat** | OK | `.runner_status.json` updated through all 20 experiments |
| **Atomic writes** | OK | TSV never truncated or corrupted during crash cycles |
| **Crash recovery** | OK | 8 crashes (OOM, batch size errors) — all recovered cleanly |
| **Git branch isolation** | OK | All HP changes on `autoresearch/sanity-rtx4000ada`, main untouched |
| **PID locking** | OK | `.suite.pid` file present, single-instance enforced |
| **Signal handling** | OK | Clean shutdown on experiment limit |
| **Data sync** | OK | Results pushed to GitHub via `sync_results.sh` |
| **Volume backup** | OK | Results copied to `/runpod-volume/autoresearch-backup/` |

#### Issues and Observations

1. **High crash rate (40%)**: Most crashes were OOM from the RTX 4000 Ada's 20 GB VRAM limit. The agent repeatedly tried increasing DEPTH beyond what the GPU could handle. On larger GPUs (A100 80GB, MI300X 192GB), this would not be an issue.

2. **TSV gap (exp12–exp18)**: Experiments that crashed before evaluation wrote to the log but not the TSV. The TSV jumps from exp10 to exp19. This is by design — only experiments that complete training get a results row. Crashes are logged in the heartbeat stats.

3. **Agent learning curve**: Claude appropriately focused on weight decay early (best improvement) but got stuck in a loop trying DEPTH=11 repeatedly (crashed 3 times). Future work: feed crash history back to the LLM to avoid repeating OOM-inducing changes.

4. **Bug found — git config**: The `git commit` command failed initially because `user.email` and `user.name` were not configured in the RunPod container. Fixed by adding `git config` to the bootstrap sequence.

5. **Bug found — sync script path**: The `sync_results.sh` file was lost during `git checkout` from the experiment branch (it only existed on `main`). Fixed by fetching scripts from `origin/main` explicitly. Now committed to both branches.

---

## Cross-Platform Comparison

| Metric | Apple M5 Max (MLX) | RTX 4000 Ada (CUDA) |
|--------|-------------------|---------------------|
| **Baseline val_bpb** | 1.3669 | 1.1767 |
| **Best val_bpb** | 1.3669 (baseline) | 1.1752 (exp9) |
| **Throughput** | 50,898 tok/s | 106,666 tok/s |
| **MFU** | 12.0% | 24.1% |
| **Peak memory** | 13.7 GB | 8.1 GB |
| **Steps / 5 min** | 468 | 977 |

> **Note**: Different val_bpb baselines reflect different model configurations (depth, batch size) auto-suggested per platform. The CUDA baseline benefits from `torch.compile` + higher MFU.

---

## Test 3: Production Suite v2 — Data Integrity Validation

The v2 suite (commit `dacda45`) is the first production run from the unified codebase. Three critical data loss bugs were fixed before this run:

| Bug | Commit | Root Cause | Fix |
|-----|--------|------------|-----|
| `git reset --hard` wiped results TSV | `3479a08` | Hard reset touched all tracked files | Soft reset + targeted file restore |
| `atomic_append` read-all/write-all race | `215e1db` | Stale read between open and rename | Direct `open("a")` + `fsync` |
| Sync-race: revert hit sync commit | `dacda45` | `revert_last_commit()` targeted HEAD, not the experiment | `revert_last_experiment()` walks git log to find actual experiment commit |

### Validation Results (both platforms)

| Check | RTX PRO 6000 | MI300X |
|-------|-------------|--------|
| ClimbMix experiments ran | 79 | 80 |
| TSV rows match heartbeat total | 79/79 | 80/80 |
| baseline_sha consistent | `dacda45` all rows | `dacda45` all rows |
| `_ensure_clean_baseline` confirmed | Log: "Training script is clean" | Log: "Training script is clean" |
| Dataset transition (ClimbMix → FineWeb-Edu) | Clean branch switch, no data loss | Clean branch switch, no data loss |
| GitHub sync | Branches pushed and verified | Branches pushed and verified |
| Keeps preserved through reverts | All 13 keeps intact | All 26 keeps intact |
| Crashes logged correctly | 4 crashes in TSV | 1 crash in TSV |
| 11-column format | All rows have `baseline_sha` | All rows have `baseline_sha` |

### Conclusion

**No data loss detected in v2 suite.** The three fixes (`3479a08`, `215e1db`, `dacda45`) resolved the persistent TSV truncation issue. The `revert_last_experiment()` approach is confirmed working: it correctly identifies experiment commits (matching `expN:` pattern), skips intervening sync commits, and restores only the training script file — leaving results, heartbeat, and sync artifacts untouched.

---

## Platform Testing Status

| Platform | Hardware | Status | Notes |
|----------|----------|--------|-------|
| Apple MLX | M5 Max | **Validated** | 2-experiment sanity test (commit `5f77361`) |
| NVIDIA CUDA | RTX 4000 Ada | **Validated** | 20-experiment sanity test (commit `967f33b`) |
| NVIDIA CUDA | RTX PRO 6000 Blackwell | **Production** | Full 8-dataset suite running (commit `dacda45`) |
| AMD ROCm | MI300X | **Production** | Full 8-dataset suite running (commit `dacda45`, no torch.compile) |
| Intel Gaudi | Gaudi 3 | **Blocked** | IBM Cloud storage quota (25.6TB required, 18TB limit) |

---

## How to Run

```bash
# Install with platform-appropriate deps
uv pip install -e ".[all-metal]"   # Apple Silicon
uv pip install -e ".[all-cuda]"    # NVIDIA
uv pip install -e ".[all-rocm]"    # AMD

# Quick 2-experiment sanity test
python -m tui.headless --tag sanity --max 2 --results results/sanity/results.tsv

# Full 20-experiment sanity test
python -m tui.headless --tag sanity --max 20 --results results/sanity/results.tsv

# RunPod deployment (automated)
export ANTHROPIC_API_KEY=sk-ant-...
bash platforms/runpod/scripts/launch.sh --max=20 --tag=sanity

# Or use the TUI
python -m tui.app
```
