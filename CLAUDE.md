# Autoresearch-Unified — Project Memory

## Repository
- **GitHub**: https://github.com/elementalcollision/autoresearch-unified
- **Local path**: `/Users/dave/Claude_Primary/autoresearch-unified`
- **Python venv**: `.venv` (Python 3.14 locally — only used for RunPod SDK calls, not training)

## RunPod Deployment — What Works and What Doesn't

### SSH Access
- **Direct SSH** (`ssh root@<ip> -p <port>`) works when pod has `22/tcp` in port config
- **RunPod SSH proxy** requires `-tt` flag: `ssh -tt <podid>-<hostid>@ssh.runpod.io -i ~/.ssh/id_ed25519`
- **Piping commands via stdin does NOT work** with RunPod SSH proxy — you get "PTY not supported"
- **Use paramiko** `invoke_shell()` for programmatic access through the proxy — this is the ONLY reliable method from Claude Code
- **runpodctl** (brew install): `ssh info` command exists but does NOT support remote exec

### paramiko Pattern (RELIABLE)
```python
import paramiko, time, re
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname="ssh.runpod.io", username="<podid>-<hostid>@ssh.runpod.io", timeout=30)
chan = client.invoke_shell(width=300, height=50)
time.sleep(2)
if chan.recv_ready(): chan.recv(65536)  # clear banner
chan.send("<commands>\n")
# Poll for sentinel string in output
# Clean ANSI: re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', output)
```

### Pod Provisioning
- **Valid RunPod PyTorch images** (as of March 2026):
  - `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (BEST for Blackwell)
  - `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (for older drivers)
  - `runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
  - `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`
  - `runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04` (for MI300x)
- **INVALID images** (do NOT exist): `runpod/pytorch:2.6.0-py3.11-cuda12.8.1-devel-ubuntu22.04`
- **RTX 4090 availability**: Frequently at capacity — have fallback GPU list ready
- **Image pull time**: Large images (~15GB) can take 2-5 min to pull. `uptimeSeconds: 0` with `desiredStatus: RUNNING` means still pulling.
- **Hung GPUs**: Some machines have GPUs in error state. Test `nvidia-smi` early. If it hangs, terminate and reprovision.

### PyTorch Version Compatibility
- `pyproject.toml` requires `torch>=2.6.0`
- RunPod image PyTorch 2.4.x does NOT satisfy this — `pip install -e ".[all-cuda]"` will upgrade torch, which **breaks CUDA driver compatibility** (e.g., 2.11.0 needs newer driver than what's on the image)
- **Fix**: Either use an image with PyTorch >= 2.6.0, or pin torch BEFORE installing: `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124`
- **Best approach**: Use `runpod/pytorch:2.8.0-*` image which ships 2.8.0 — no conflicts

### Cron / Background Sync
- **RunPod containers do NOT have `crontab`** — the command doesn't exist
- Use a background bash loop instead:
  ```bash
  nohup bash -c 'while true; do sleep 600; bash sync_results.sh; done' &
  ```
- Store PID for cleanup

### Git Auth on Pods
- Pods have no GitHub credentials by default
- Set up via: `git remote set-url origin https://x-access-token:<TOKEN>@github.com/...`
- Configure: `git config user.email "autoresearch@runpod.local"` and `user.name`

### Data Prep
- `prepare.py` is **hardcoded to ClimbMix** — no `--dataset` arg
- Downloads from HuggingFace, trains BPE tokenizer
- ~3-5 min on fast instance
- Data goes to `$AUTORESEARCH_CACHE_DIR` (default: `~/.cache/autoresearch`)

## Critical Bugs Found and Fixed

### 1. `git reset --hard` Data Loss (commit `3479a08`)
- **Symptom**: Results TSV only had 1 of N experiment rows after a run
- **Root cause**: `revert_last_commit()` used `git reset --hard HEAD~1` which reset ALL tracked files. When the sync script committed the results TSV to git, a subsequent discard/crash revert would nuke the results back to the last-synced state.
- **Fix**: Replaced with soft reset + targeted file restore — only revert files changed in the experiment commit (i.e., the training script). Results, heartbeat, and sync artifacts are untouched.

### 2. OOM on Sub-24GB GPUs (commit `e6e70b2`)
- RTX 4000 Ada (20GB) was crashing on larger batch sizes / deeper models
- `suggest_hyperparameters()` now adjusts based on VRAM tier

## GPU FLOPS Table (`backends/registry.py`)
Key entries:
- RTX 4090: 330e12
- RTX 4000 Ada: 105e12
- RTX PRO 6000 (Blackwell): 380e12
- L40S: 362e12
- A100: 312e12
- H100: 756e12
- MI300X: 1307e12 (ROCm)
- Gaudi 3: 1835e12 (HPU)

## Sanity Test Results

### RTX 4000 Ada (20GB) — 20 experiments
- Best val_bpb: 1.175187 (exp9: weight_decay=0.05)
- 4 kept, 8 discarded, 8 crashes (OOM)
- ~107K tok/sec, 24% MFU
- Branch: `autoresearch/sanity-rtx4000ada`

### RTX PRO 6000 Blackwell (102GB) — 5 experiments (v1, data lost)
- Best val_bpb: 1.077628 (baseline)
- ~251K tok/sec, 28.3% MFU, 17.1GB VRAM
- Data lost due to git reset --hard bug (fixed in 3479a08)
- v2 rerun in progress

## Architecture Notes
- **Training scripts stay separate** — ~570 lines of deeply interleaved platform-specific code; template method pattern would be fragile
- **Optimizer files stay as-is** — identical algorithm but different `@torch.compile` decorators
- **ROCm resilience module is universal** — atomic writes, heartbeat, signal handlers
- **`backends/registry.py`** — single registration point for new platforms (training script, FLOPS, display name)
- **Cross-dataset baseline isolation** — `_ensure_clean_baseline()` resets train_xxx.py to `main` branch state before each new dataset run
- **Results format**: 11-column TSV (exp, description, val_bpb, peak_mem_gb, tok_sec, mfu, steps, status, notes, gpu_name, baseline_sha)

## Key File Paths
- Training scripts: `platforms/{cuda,rocm,metal,gaudi}/train_*.py`
- Backend detection: `backends/__init__.py` → `detect_backend()`
- Platform registry: `backends/registry.py` → `PLATFORMS` dict
- Orchestrator: `tui/orchestrator.py` → `ExperimentOrchestrator`
- Headless runner: `tui/headless.py` → `python -m tui.headless --tag X --max N --results path.tsv`
- Results: `tui/results.py` (11-col TSV with baseline_sha)
- Resilience: `tui/resilience.py` (atomic writes, heartbeat, signal handlers)
- Git operations: `tui/git_manager.py` (baseline tracking, targeted revert)
- RunPod sync: `platforms/runpod/scripts/sync_results.sh`
- RunPod launch: `platforms/runpod/scripts/launch.sh`
- RunPod stop: `platforms/runpod/scripts/stop.sh`

## Source Repos (archived as branches)
- Metal: `archive/metal` (from `autoresearch`)
- CUDA: `archive/cuda` (from `autoresearch-cuda`)
- ROCm: `archive/rocm` (from `autoresearch-rocm`)
- Gaudi: `archive/gaudi` (from `autoresearch-gaudi`)
