#!/bin/bash
# =============================================================================
# autoresearch-unified — One-Shot RunPod Bootstrap
#
# Single command to go from bare pod to running experiments.
# Handles CUDA and ROCm backends. Target: ready in <20 minutes.
#
# Usage:
#   # Minimal (auto-detects everything):
#   export ANTHROPIC_API_KEY=sk-ant-...
#   curl -sL https://raw.githubusercontent.com/elementalcollision/autoresearch-unified/main/platforms/runpod/scripts/launch.sh | bash
#
#   # Or from cloned repo:
#   bash platforms/runpod/scripts/launch.sh
#
#   # With options:
#   bash launch.sh --max=5 --tag=sanity-test --gh-token=gho_xxx
#
#   # Skip agent (bootstrap only):
#   bash launch.sh --skip-agent
#
# =============================================================================
set -euo pipefail

# ── Configuration (env vars or CLI args) ─────────────────────
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-30}
NUM_SHARDS=${NUM_SHARDS:-10}
TAG=${TAG:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}
DATASET=${DATASET:-climbmix}
REPO_URL="https://github.com/elementalcollision/autoresearch-unified.git"
WORKSPACE="/workspace/autoresearch-unified"
SKIP_AGENT=false
GH_TOKEN="${GH_TOKEN:-}"

for arg in "$@"; do
    case $arg in
        --help)
            cat <<HELP
Usage: bash launch.sh [OPTIONS]

Options:
  --max=N            Max experiments (default: 30)
  --tag=TAG          Run tag (default: date)
  --dataset=NAME     Dataset (default: climbmix)
  --shards=N         Data shards (default: 10)
  --model=MODEL      Model override (e.g. 'qwen/qwen3.5-397b-a17b' for OpenRouter)
  --gh-token=TOKEN   GitHub token for result sync
  --skip-agent       Stop after install + data prep (no experiments)
  --help             Show this help

Environment variables:
  ANTHROPIC_API_KEY    API key for Anthropic (direct)
  OPENROUTER_API_KEY   API key for OpenRouter (alternative to Anthropic)
  OPENROUTER_MODEL     Model ID for OpenRouter (or use --model)
  GH_TOKEN             GitHub token for result sync (or use --gh-token)
  MAX_EXPERIMENTS      Same as --max
  TAG                  Same as --tag
HELP
            exit 0 ;;
        --skip-agent)       SKIP_AGENT=true ;;
        --max=*)            MAX_EXPERIMENTS="${arg#*=}" ;;
        --shards=*)         NUM_SHARDS="${arg#*=}" ;;
        --tag=*)            TAG="${arg#*=}" ;;
        --dataset=*)        DATASET="${arg#*=}" ;;
        --model=*)          OPENROUTER_MODEL="${arg#*=}"; export OPENROUTER_MODEL ;;
        --gh-token=*)       GH_TOKEN="${arg#*=}" ;;
        *)                  echo "Unknown: $arg (try --help)"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'
T0=$(date +%s)

phase()   { echo -e "\n${BLUE}━━━ Phase $1: $2 ━━━${NC}"; }
info()    { echo -e "  ${BLUE}$1${NC}"; }
success() { echo -e "  ${GREEN}✓ $1${NC}"; }
warn()    { echo -e "  ${YELLOW}⚠ $1${NC}"; }
fail_exit() { echo -e "\n  ${RED}FATAL: $1${NC}"; exit 1; }
elapsed() {
    local d=$(( $(date +%s) - T0 ))
    echo -e "  ${YELLOW}Elapsed: $((d/60))m $((d%60))s${NC}"
}

# =============================================================================
# Phase 0: Detect Platform (CUDA vs ROCm)  ~10s
# =============================================================================
phase "0" "Platform detection"

BACKEND="unknown"
GPU_NAME="unknown"

# CUDA path
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | xargs)
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | xargs)
    if [ -n "$GPU_NAME" ] && [ "$GPU_NAME" != "" ]; then
        BACKEND="cuda"
        success "NVIDIA: $GPU_NAME ($GPU_VRAM) — driver $DRIVER"
    fi
fi

# ROCm path
if [ "$BACKEND" = "unknown" ] && command -v rocm-smi &>/dev/null; then
    GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card series" | head -1 | sed 's/.*: //' | xargs)
    ROCM_VER=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    if [ -n "$GPU_NAME" ]; then
        # Detect ROCm 6.x vs 7.x
        MAJOR=$(echo "$ROCM_VER" | cut -d. -f1)
        if [ "$MAJOR" -ge 7 ] 2>/dev/null; then
            BACKEND="rocm7"
        else
            BACKEND="rocm"
        fi
        success "AMD: $GPU_NAME — ROCm $ROCM_VER (backend: $BACKEND)"
    fi
fi

[ "$BACKEND" = "unknown" ] && fail_exit "No NVIDIA or AMD GPU detected"

# Check PyTorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
if [ "$TORCH_VER" = "none" ]; then
    warn "PyTorch not pre-installed — pip will install it"
else
    success "PyTorch: $TORCH_VER"
    # Verify GPU is accessible from PyTorch
    GPU_OK=$(python3 -c "import torch; print('ok' if torch.cuda.is_available() else 'fail')" 2>/dev/null || echo "fail")
    if [ "$GPU_OK" = "fail" ]; then
        warn "PyTorch cannot see GPU — may need version pin (see CLAUDE.md)"
    else
        success "PyTorch GPU access: verified"
    fi
fi

# Check API key (Anthropic direct OR OpenRouter)
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    success "OPENROUTER_API_KEY: set (${#OPENROUTER_API_KEY} chars)"
    [ -n "${OPENROUTER_MODEL:-}" ] && success "OPENROUTER_MODEL: $OPENROUTER_MODEL"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    success "ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)"
else
    warn "No API key set — agent will not run"
    warn "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY"
    if ! $SKIP_AGENT; then
        echo -n "  Enter Anthropic API key (or press Enter to skip agent): "
        read -r ANTHROPIC_API_KEY
        [ -z "$ANTHROPIC_API_KEY" ] && SKIP_AGENT=true || export ANTHROPIC_API_KEY
    fi
fi

info "Backend: $BACKEND | GPU: $GPU_NAME | Tag: $TAG | Max: $MAX_EXPERIMENTS"
elapsed

# =============================================================================
# Phase 1: Clone & Install  ~2-5 min
# =============================================================================
phase "1" "Repository & dependencies"

if [ -d "$WORKSPACE/.git" ]; then
    info "Repo exists, pulling latest..."
    cd "$WORKSPACE"
    git checkout main 2>/dev/null || true
    git pull --ff-only origin main 2>/dev/null || warn "Pull failed (may be on experiment branch)"
else
    info "Cloning $REPO_URL..."
    git clone "$REPO_URL" "$WORKSPACE"
    cd "$WORKSPACE"
fi
success "Repo: $(git log --oneline -1)"

# Git config for experiment commits
git config user.email "autoresearch@runpod.local"
git config user.name "autoresearch-bot"
git config push.autoSetupRemote true

# GitHub auth for sync (if token provided)
if [ -n "$GH_TOKEN" ]; then
    git remote set-url origin "https://x-access-token:${GH_TOKEN}@github.com/elementalcollision/autoresearch-unified.git"
    success "GitHub sync: configured"
fi

# Install deps — pick the right extra based on backend
INSTALL_START=$(date +%s)
if [ "$BACKEND" = "cuda" ]; then
    # Check if torch upgrade would break things
    if [ "$TORCH_VER" != "none" ]; then
        TORCH_MAJOR=$(echo "$TORCH_VER" | cut -d. -f1-2 | sed 's/[^0-9.]//g')
        # If pre-installed torch < 2.6, constrain to avoid upgrade
        if python3 -c "exit(0 if float('$TORCH_MAJOR') < 2.6 else 1)" 2>/dev/null; then
            warn "Pre-installed PyTorch $TORCH_VER < 2.6.0 — installing deps without torch upgrade"
            pip install -e ".[agent,analysis]" 2>&1 | tail -3
        else
            pip install -e ".[all-cuda]" 2>&1 | tail -3
        fi
    else
        pip install -e ".[all-cuda]" 2>&1 | tail -3
    fi
elif [ "$BACKEND" = "rocm" ] || [ "$BACKEND" = "rocm7" ]; then
    pip install -e ".[all-rocm]" 2>&1 | tail -3
fi
success "Install: $(($(date +%s) - INSTALL_START))s"

# Set environment
export AUTORESEARCH_BACKEND="$BACKEND"
export AUTORESEARCH_CACHE_DIR="/workspace/.cache/autoresearch"
export PYTHONPATH="$WORKSPACE"
mkdir -p "$AUTORESEARCH_CACHE_DIR"

# Verify backend detection
DETECTED=$(python3 -c "
import sys; sys.path.insert(0, '.')
from backends import detect_backend, get_hardware_info, get_peak_flops
hw = get_hardware_info()
flops = get_peak_flops(hw)
backend = detect_backend()
print(f'{backend}|{hw.get(\"chip_name\",\"?\")[:40]}|{flops:.2e}')
" 2>/dev/null || echo "error|detection failed|0")

IFS='|' read -r DET_BACKEND DET_CHIP DET_FLOPS <<< "$DETECTED"
if [ "$DET_BACKEND" = "error" ]; then
    warn "Backend detection failed — $DET_CHIP"
else
    success "Detected: $DET_BACKEND | $DET_CHIP | $DET_FLOPS FLOPS"
fi

elapsed

# =============================================================================
# Phase 2: Prepare Data  ~3-5 min (skipped if cached)
# =============================================================================
phase "2" "Data preparation ($DATASET)"

DATA_DIR="$AUTORESEARCH_CACHE_DIR/data"
TOK_DIR="$AUTORESEARCH_CACHE_DIR/tokenizer"

if [ -d "$DATA_DIR" ] && [ "$(ls "$DATA_DIR"/shard_*.parquet 2>/dev/null | wc -l)" -ge "$NUM_SHARDS" ] && [ -f "$TOK_DIR/tokenizer.bin" ]; then
    SHARD_COUNT=$(ls "$DATA_DIR"/shard_*.parquet | wc -l)
    success "Data cached: $SHARD_COUNT shards + tokenizer (skipping download)"
else
    info "Downloading $NUM_SHARDS shards from HuggingFace..."
    PREP_START=$(date +%s)
    python prepare.py --num-shards="$NUM_SHARDS"
    success "Data prep: $(($(date +%s) - PREP_START))s"
fi

elapsed

# =============================================================================
# Phase 3: Launch Agent  ~5 min baseline + N*5 min experiments
# =============================================================================
if $SKIP_AGENT; then
    phase "3" "SKIPPED (--skip-agent)"
    echo
    echo "  Pod is ready. To launch agent manually:"
    echo "    cd $WORKSPACE"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    python -m tui.headless --tag $TAG --max $MAX_EXPERIMENTS --results results_${TAG}.tsv --dataset $DATASET"
    echo
    elapsed
    exit 0
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENROUTER_API_KEY:-}" ]; then
    phase "3" "SKIPPED (no API key)"
    warn "Set ANTHROPIC_API_KEY or OPENROUTER_API_KEY to run experiments"
    elapsed
    exit 0
fi

phase "3" "Autonomous agent ($MAX_EXPERIMENTS experiments)"

cd "$WORKSPACE"

# Determine results file and training script
RESULTS_FILE="results_${TAG}.tsv"
TRAINING_SCRIPT=$(python3 -c "from backends.registry import get_training_script; print(get_training_script('$BACKEND'))" 2>/dev/null || echo "")
[ -n "$TRAINING_SCRIPT" ] && info "Training script: $TRAINING_SCRIPT"

# Create experiment branch
SAFE_GPU=$(echo "$GPU_NAME" | tr ' /' '-' | tr '[:upper:]' '[:lower:]' | head -c 30)
BRANCH="autoresearch/${TAG}-${SAFE_GPU}"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH" 2>/dev/null || true
if [ -n "$GH_TOKEN" ]; then
    git push -u origin "$BRANCH" 2>/dev/null || warn "Initial push failed"
fi

# Set up background sync (RunPod has no crontab)
SYNC_SCRIPT="$WORKSPACE/platforms/runpod/scripts/sync_results.sh"
if [ -f "$SYNC_SCRIPT" ]; then
    chmod +x "$SYNC_SCRIPT"
    export AUTORESEARCH_WORKSPACE="$WORKSPACE"
    (
        while true; do
            sleep 600
            bash "$SYNC_SCRIPT" >> "$WORKSPACE/sync.log" 2>&1
        done
    ) &
    SYNC_PID=$!
    success "Background sync: PID $SYNC_PID (every 10 min)"
fi

info "Starting headless runner..."
info "  Results: $RESULTS_FILE"
info "  Branch: $BRANCH"
echo

# Build model argument if OPENROUTER_MODEL is set
MODEL_ARG=""
if [ -n "${OPENROUTER_MODEL:-}" ]; then
    MODEL_ARG="--model $OPENROUTER_MODEL"
fi

python -m tui.headless \
    --tag "$TAG" \
    --max "$MAX_EXPERIMENTS" \
    --results "$RESULTS_FILE" \
    --dataset "$DATASET" \
    $MODEL_ARG \
    2>&1 | tee /tmp/agent_run.log

# Final sync
[ -f "$SYNC_SCRIPT" ] && bash "$SYNC_SCRIPT" 2>/dev/null || true

# Kill sync loop
[ -n "${SYNC_PID:-}" ] && kill "$SYNC_PID" 2>/dev/null || true

# =============================================================================
# Summary
# =============================================================================
echo
echo -e "${BLUE}━━━ Complete ━━━${NC}"
elapsed

if [ -f "$WORKSPACE/$RESULTS_FILE" ]; then
    TOTAL=$(($(wc -l < "$WORKSPACE/$RESULTS_FILE") - 1))
    BEST=$(tail -n +2 "$WORKSPACE/$RESULTS_FILE" | awk -F'\t' '$3 > 0 {print $3}' | sort -n | head -1)
    echo -e "  ${GREEN}Results: $TOTAL experiments, best val_bpb=$BEST${NC}"
    echo "  File: $RESULTS_FILE"
fi

echo
echo "  Next steps:"
echo "    - View results:  column -t -s\$'\\t' $RESULTS_FILE"
echo "    - Resume:        bash platforms/runpod/scripts/launch.sh --max=100 --tag=$TAG"
echo "    - Graceful stop: bash platforms/runpod/scripts/stop.sh"
echo "    - Stop pod to save credits!"
echo
