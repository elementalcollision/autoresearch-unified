# Autoresearch-Unified — Project Memory

## Repository
- **GitHub**: https://github.com/elementalcollision/autoresearch-unified
- **Local path**: `/Users/dave/Claude_Primary/autoresearch-unified`
- **Python venv**: `.venv` (Python 3.14 locally — only used for RunPod SDK calls, not training)

## Quick Deploy (Target: <20 min pod-to-experiments)

One-liner from any RunPod pod (CUDA or ROCm):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
export GH_TOKEN=gho_...   # optional, for result sync to GitHub
curl -sL https://raw.githubusercontent.com/elementalcollision/autoresearch-unified/main/platforms/runpod/scripts/launch.sh | bash -s -- --max=5 --tag=sanity
```

Or if already cloned:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
bash platforms/runpod/scripts/launch.sh --max=5 --tag=sanity --gh-token=gho_...
```

The script auto-detects CUDA vs ROCm, handles PyTorch version conflicts, skips data prep if cached, and sets up background sync.

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

### 2. `atomic_append` Read-All/Write-All Data Loss (commit `215e1db`)
- **Symptom**: Same as above — only the last experiment row survived
- **Root cause**: The original `atomic_append` read the entire file, appended a line, then did an atomic write (tmp+rename). If any process touched the file between the read and the rename, the stale read + new line would overwrite the newer content.
- **Fix**: Replaced with direct `open("a")` + `fsync`. Safe because the orchestrator is single-threaded. Partial writes from crashes are caught by `validate_results_tsv()` on next startup.

### 3. Sync-Race Data Loss in `revert_last_commit` (commit `dacda45`)
- **Symptom**: Log showed "Total: 80 experiments" per dataset, but TSV had 8-21 rows. All crash/discard rows lost.
- **Root cause**: Background sync script commits results TSV every 10 min. When experiment N crashes, `revert_last_commit()` reverted HEAD — but HEAD might be a sync commit (not the experiment commit). The sync commit's diff included `results.tsv`, so the revert restored the TSV to a pre-sync state, wiping all rows added since the last sync.
- **Evidence**: v1 suite on RTX PRO 6000 — ClimbMix: 80 experiments ran, 8 TSV rows. FineWeb-Edu: 80 ran, 20 rows. Cosmopedia: 80 ran, 14 rows.
- **Fix**: Replaced `revert_last_commit()` with `revert_last_experiment()`. Walks `git log` to find the actual experiment commit (matching `expN:` pattern), skips sync commits entirely, restores only the experiment's files from its parent commit, and creates a clean revert commit.
- **Validated**: v2 suite — 30-min check showed 10/10 experiments preserved in TSV (baseline + 3 keeps + 3 crashes + 3 discards all present).

### 3. OOM on Sub-24GB GPUs (commit `e6e70b2`)
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

### RTX PRO 6000 Blackwell (102GB) — v2 suite (validated, running)
- Best val_bpb: **1.072617** (exp7: MATRIX_LR=0.03) after 10 experiments
- ~253K tok/sec, 28.3% MFU, 17.1GB VRAM (depth=10)
- **All rows persisted** — data loss fix confirmed (10 TSV rows = 10 heartbeat total)
- baseline_sha `dacda45` correctly populated on all rows
- Running full 8-dataset suite (80 exp/dataset, ~53 hours)
- Branch: `autoresearch/suite-nvidia-rtx-pro-6000-black-v2`

### MI300X (206GB) — v1 suite (launching)
- Backend: ROCm 6.2 → `platforms/rocm/train_rocm.py`
- 304 compute units, gfx942 CDNA3
- Peak FLOPS: 1.31e+15
- Running full 8-dataset suite (80 exp/dataset)
- Branch: `autoresearch/suite-mi300x`
- Cost: $0.50/hr (secure)

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

## Active Deployment Environments

### RTX PRO 6000 Blackwell (CUDA)
- **RunPod Pod ID**: `puuaenj86l7hzt` | SSH: `puuaenj86l7hzt-64411d40@ssh.runpod.io` (paramiko only)
- **GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition — 102 GB VRAM
- **Driver**: 570.195.03 | CUDA 12.8
- **PyTorch**: 2.8.0.dev20250319+cu128
- **Image**: `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`
- **Backend**: `cuda` → `platforms/cuda/train_cuda.py`
- **Peak FLOPS**: 380e12 (registry key: `"rtx pro 6000"`)
- **Baseline SHA**: `dacda45`
- **Suite branch**: `autoresearch/suite-nvidia-rtx-pro-6000-black-v2`
- **Per-dataset branches**: `autoresearch/rtxpro6000-mar24-v2-<dataset>`
- **Cost**: $1.69/hr on-demand
- **Status**: Running full 8-dataset suite (80 exp/dataset)

### MI300X (ROCm 6.x) — v2
- **RunPod Pod ID**: `drrcdswdfur3e7` | SSH: `root@213.173.96.53 -p 10118`
- **GPU**: AMD Instinct MI300X OAM — 206 GB VRAM (192 GB usable)
- **GCN Architecture**: gfx942:sramecc+:xnack- (CDNA3)
- **GPU Compute Units**: 304
- **ROCm**: 6.1.0-82 (system) / 6.3.42131 (PyTorch HIP)
- **PyTorch**: 2.9.1+rocm6.3 recommended (**CRITICAL**: 2.5.1+rocm6.2 produces NaN on first step — DO NOT USE; v2 suite ran on 2.8.0+rocm6.3)
- **Image**: `runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04` (base — install PyTorch separately)
- **Container disk**: 80 GB (40GB was too small for PyTorch 2.8 ROCm wheel ~5GB)
- **Backend**: `rocm` → `platforms/rocm/train_rocm.py`
- **torch.compile**: Tiered fallback chain (Inductor → aot_eager → eager). On 2.8.0 the Inductor backend crashes; upgrade to 2.9.1 expected to fix. Env vars: `AUTORESEARCH_COMPILE_BACKEND`, `AUTORESEARCH_COMPILE_MODE`, `AUTORESEARCH_NO_COMPILE`
- **Peak FLOPS**: 1.31e+15 (registry key: `"mi300x"`)
- **Baseline SHA**: `dacda45`
- **Suite branch**: `autoresearch/suite-mi300x-v2`
- **Per-dataset branches**: `autoresearch/mi300x-mar24-v2-<dataset>`
- **Cost**: $0.50/hr secure
- **Status**: Running full 8-dataset suite (80 exp/dataset)

### ROCm Image Selection for MI300X
- `runpod/pytorch:2.4.0-py3.10-rocm6.1.0-ubuntu22.04` — WORKS as base image (tested, confirmed)
- `runpod/pytorch:2.6.0-py3.12-rocm6.2.4-ubuntu22.04` — UNTESTED (image pull hung on provisioning)
- **MUST use PyTorch ≥2.8.0+rocm6.3** — 2.5.1+rocm6.2 produces NaN on first training step
- Install separately: `pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.3`
- Container disk must be ≥80GB (5GB wheel + existing packages + data)
- torch.compile now has a tiered fallback chain (Inductor → aot_eager → eager) — `AUTORESEARCH_NO_COMPILE=1` no longer required but still supported
- Available PyTorch ROCm versions: rocm6.2 (max 2.5.1), rocm6.3 (up to 2.9.1), rocm6.4 (up to 2.9.1)

### MI300X Known Issues
1. **NaN on first step with PyTorch 2.5.1+rocm6.2**: Model initialization produces NaN loss immediately. Fixed by upgrading to 2.8.0+rocm6.3.
2. **torch.compile Inductor crash (PyTorch 2.8.0)**: `AssertionError: wrong number of dimensions` in compiled backward pass. Root cause: Inductor shape inference bug with CDNA3 architecture. **Remediation**: Upgrade to PyTorch 2.9.1+rocm6.3 (expected fix). Fallback chain added in `fix/rocm-torch-compile-mi300x` — tries Inductor, then aot_eager, then eager. Env var `AUTORESEARCH_NO_COMPILE=1` still supported as manual override.
3. **Low MFU without compile**: ~8% MFU (7.1% measured in v2 suite) vs 33.7% on CUDA with compile enabled. The 4.7x gap is entirely attributable to torch.compile being disabled. Target: 15-25% MFU after re-enabling compile with PyTorch 2.9.1.
4. **Container disk 40GB too small**: PyTorch ROCm wheel is ~5GB. Need ≥80GB container disk.

### Vultr
- **Account**: dave@elementalcollision.com
- **API Key**: starts with `7BDAT...`
- **Bare metal GPU plans**: NOT enabled (0 plans returned). Requires account activation via support ticket.
- **Cloud GPU plans**: 29 available (A16, A40, A100, L40S) but no GH200
- **GH200 plan ID**: `vbm-72c-480gb-gh200-gpu` (bare metal only — needs activation)

## API Keys Reference
- **RunPod**: `rpa_9ZJOH9H...` (env: RUNPOD_API_KEY)
- **Anthropic**: `sk-ant-api03-jpnc...` (env: ANTHROPIC_API_KEY)
- **GitHub token**: `gho_gizZAG3...` (for git push from pods)
- **Vultr**: `7BDAT72T...` (env: VULTR_API_KEY — bare metal not yet activated)

## Source Repos (archived as branches)
- Metal: `archive/metal` (from `autoresearch`)
- CUDA: `archive/cuda` (from `autoresearch-cuda`)
- ROCm: `archive/rocm` (from `autoresearch-rocm`)
- Gaudi: `archive/gaudi` (from `autoresearch-gaudi`)
