#!/bin/bash
# =============================================================================
# autoresearch-runpod — Launch Day Script
#
# Run this ON a RunPod GPU pod. It does everything:
#   0. Validate environment (nvidia-smi, CUDA, API key)
#   1. Clone repo + pip install
#   2. Download + prepare ClimbMix data
#   3. Run a baseline training (~5 min)
#   4. Launch autonomous agent (ClimbMix benchmark)
#
# Prerequisites:
#   - RunPod GPU pod is running (A100, H100, RTX 4090, etc.)
#   - You've SSH'd into the pod
#   - ANTHROPIC_API_KEY is set (or will be prompted)
#
# Usage:
#   # Full automated run:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash launch.sh
#
#   # Stop after baseline (no agent):
#   bash launch.sh --skip-agent
#
#   # Show help:
#   bash launch.sh --help
#
# Estimated runtime:
#   Phase 0 (validate):    ~10s
#   Phase 1 (install):     ~2-5 min
#   Phase 2 (data prep):   ~3-5 min
#   Phase 3 (baseline):    ~5 min
#   Phase 4 (agent):       ~2-8 hrs (depends on MAX_EXPERIMENTS)
#   Total without agent:   ~10-15 min
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-30}
NUM_SHARDS=${NUM_SHARDS:-10}
TAG=${TAG:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}
DATASET=${DATASET:-climbmix}
REPO_URL="https://github.com/elementalcollision/autoresearch-unified.git"
WORKSPACE="/workspace/autoresearch-unified"
SKIP_AGENT=false

# ── Parse arguments ──────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --help)
            echo "Usage: bash launch.sh [OPTIONS]"
            echo
            echo "Options:"
            echo "  --help           Show this help message"
            echo "  --skip-agent     Stop after baseline training (skip Phase 4)"
            echo "  --max=N          Max autonomous experiments (default: 30)"
            echo "  --shards=N       Number of data shards (default: 10)"
            echo "  --tag=TAG        Experiment tag (default: date, e.g. mar23)"
            echo "  --dataset=NAME   Dataset name (default: climbmix)"
            echo
            echo "Environment variables:"
            echo "  ANTHROPIC_API_KEY   Required for autonomous agent (Phase 4)"
            echo "  MAX_EXPERIMENTS     Same as --max"
            echo "  NUM_SHARDS          Same as --shards"
            echo "  TAG                 Same as --tag"
            echo "  DATASET             Same as --dataset"
            exit 0
            ;;
        --skip-agent)  SKIP_AGENT=true ;;
        --max=*)       MAX_EXPERIMENTS="${arg#*=}" ;;
        --shards=*)    NUM_SHARDS="${arg#*=}" ;;
        --tag=*)       TAG="${arg#*=}" ;;
        --dataset=*)   DATASET="${arg#*=}" ;;
        *)             echo "Unknown arg: $arg (try --help)"; exit 1 ;;
    esac
done

# ── Helpers ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

elapsed_start=$(date +%s)

phase() {
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Phase $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

elapsed() {
    local now=$(date +%s)
    local diff=$((now - elapsed_start))
    local min=$((diff / 60))
    local sec=$((diff % 60))
    echo -e "  ${YELLOW}Elapsed: ${min}m ${sec}s${NC}"
}

info()    { echo -e "  ${BLUE}$1${NC}"; }
success() { echo -e "  ${GREEN}$1${NC}"; }
warn()    { echo -e "  ${YELLOW}$1${NC}"; }

fail_exit() {
    echo -e "\n  ${RED}FATAL: $1${NC}"
    elapsed
    exit 1
}

# ── Header ───────────────────────────────────────────────────
echo
echo -e "${BOLD}  autoresearch-unified — RunPod Launch Script${NC}"
echo -e "  Estimated runtime: ~10-15 min (without agent), ~2-8 hrs (with agent)"
echo

# =============================================================================
# Phase 0: Environment Validation
# =============================================================================
phase "0" "Environment validation"

echo "  Host: $(hostname)"
echo "  OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Date: $(date)"
echo "  Tag: $TAG"
echo "  Dataset: $DATASET"
echo "  Max experiments: $MAX_EXPERIMENTS"

# Check NVIDIA GPU
echo
echo "  Checking NVIDIA GPU..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | xargs)
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | xargs)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
    success "GPU: ${GPU_COUNT}x ${GPU_NAME} (${GPU_VRAM})"
    success "Driver: ${DRIVER_VERSION}"
    if [ "$GPU_COUNT" -lt 1 ]; then
        fail_exit "No NVIDIA GPUs detected"
    fi
else
    fail_exit "nvidia-smi not found — NVIDIA drivers not installed"
fi

# Check CUDA toolkit
echo "  Checking CUDA toolkit..."
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    success "CUDA toolkit: ${CUDA_VERSION}"
elif python3 -c "import torch; print(torch.version.cuda)" &>/dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    success "CUDA (via PyTorch): ${CUDA_VERSION}"
else
    warn "CUDA toolkit not found via nvcc or PyTorch — will install with pip"
fi

# Check ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo
    warn "ANTHROPIC_API_KEY not set."
    echo -n "  Enter your Anthropic API key (or press Enter to skip agent): "
    read -r ANTHROPIC_API_KEY
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "  Proceeding without API key (baseline only, no agent)"
    else
        export ANTHROPIC_API_KEY
    fi
else
    success "ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)"
fi

# =============================================================================
# Phase 1: Clone & Install
# =============================================================================
phase "1" "Clone repository & install dependencies"

if [ -d "$WORKSPACE/.git" ]; then
    echo "  Repo already cloned at $WORKSPACE, pulling latest..."
    cd "$WORKSPACE"
    git pull --ff-only 2>/dev/null || echo "  (pull skipped — may be on experiment branch)"
else
    echo "  Cloning $REPO_URL..."
    git clone "$REPO_URL" "$WORKSPACE"
    cd "$WORKSPACE"
fi

echo "  Current commit: $(git log --oneline -1)"

# Install dependencies
echo
echo "  Installing dependencies (pip install -e '.[all-cuda]')..."
INSTALL_START=$(date +%s)
pip install -e ".[all-cuda]" 2>&1 | tail -5
INSTALL_END=$(date +%s)
success "Install complete in $((INSTALL_END - INSTALL_START))s"

# Set environment variables
export AUTORESEARCH_BACKEND=cuda
export AUTORESEARCH_CACHE_DIR=/workspace/.cache/autoresearch
export PYTHONPATH="$WORKSPACE"

mkdir -p "$AUTORESEARCH_CACHE_DIR"

info "AUTORESEARCH_BACKEND=$AUTORESEARCH_BACKEND"
info "AUTORESEARCH_CACHE_DIR=$AUTORESEARCH_CACHE_DIR"
info "PYTHONPATH=$PYTHONPATH"

# =============================================================================
# Phase 2: Prepare Data
# =============================================================================
phase "2" "Prepare ClimbMix data (${NUM_SHARDS} shards)"

cd "$WORKSPACE"

echo "  Downloading from HuggingFace + training tokenizer..."
PREP_START=$(date +%s)
python prepare.py --num-shards="$NUM_SHARDS"
PREP_END=$(date +%s)
success "Data prep complete in $((PREP_END - PREP_START))s"

# Verify data shards exist
echo "  Verifying data..."
python3 -c "
from pathlib import Path
import os

cache_dir = Path(os.environ.get('AUTORESEARCH_CACHE_DIR', str(Path.home() / '.cache' / 'autoresearch')))
data_dir = cache_dir / 'data'
tok_dir = cache_dir / 'tokenizer'

shards = sorted(data_dir.glob('shard_*.parquet'))
tok_exists = (tok_dir / 'tokenizer.bin').exists()

print(f'  Shards: {len(shards)}')
print(f'  Tokenizer: {\"OK\" if tok_exists else \"MISSING\"}')

if not shards:
    raise SystemExit('  ERROR: No shards found')
if not tok_exists:
    raise SystemExit('  ERROR: Tokenizer missing')
" || fail_exit "Data verification failed"

# =============================================================================
# Phase 3: Baseline Training
# =============================================================================
phase "3" "Baseline training (single GPU, ~5 min)"

cd "$WORKSPACE"

echo "  Running train.py..."
echo "  (first run may include torch.compile warmup — expect slow start)"
TRAIN_START=$(date +%s)
python train.py 2>&1 | tee /tmp/baseline_run.log | tail -30
TRAIN_END=$(date +%s)
success "Training complete in $((TRAIN_END - TRAIN_START))s"

# Extract key metrics from baseline
echo
echo "  Baseline results:"
grep -E "^(val_bpb|training_seconds|peak_vram_mb|mfu_percent|tok_sec|num_steps|depth|chip):" /tmp/baseline_run.log 2>/dev/null | sed 's/^/    /' || echo "    (could not parse — check /tmp/baseline_run.log)"

elapsed

if $SKIP_AGENT; then
    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  --skip-agent: stopping after baseline${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo "  To launch the agent later:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    cd $WORKSPACE"
    echo "    python -m tui.headless --tag $TAG --max $MAX_EXPERIMENTS --results results/$DATASET/results.tsv --dataset $DATASET"
    echo
    exit 0
fi

# =============================================================================
# Phase 4: Autonomous Agent
# =============================================================================
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    phase "4" "Autonomous agent (SKIPPED — no API key)"
    warn "Set ANTHROPIC_API_KEY and rerun, or launch manually:"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    cd $WORKSPACE"
    echo "    python -m tui.headless --tag $TAG --max $MAX_EXPERIMENTS --results results/$DATASET/results.tsv --dataset $DATASET"
else
    phase "4" "Autonomous agent — $DATASET benchmark"

    cd "$WORKSPACE"

    echo "  Dataset: $DATASET"
    echo "  Tag: $TAG"
    echo "  Max experiments: $MAX_EXPERIMENTS"
    echo "  Results: results/$DATASET/results.tsv"
    echo

    # Ensure results directory exists
    mkdir -p "results/$DATASET"

    echo "  Starting headless agent..."
    echo "  (Ctrl+C to stop gracefully)"
    echo

    python -m tui.headless \
        --tag "$TAG" \
        --max "$MAX_EXPERIMENTS" \
        --results "results/$DATASET/results.tsv" \
        --dataset "$DATASET" \
        2>&1 | tee /tmp/agent_run.log

    echo
    echo "  Agent run complete."
    echo "  Results: results/$DATASET/results.tsv"
    echo "  Full log: /tmp/agent_run.log"
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Launch Complete${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
elapsed

if [ -f "$WORKSPACE/results/$DATASET/results.tsv" ]; then
    echo
    echo "  Results summary:"
    head -1 "$WORKSPACE/results/$DATASET/results.tsv"
    tail -n +2 "$WORKSPACE/results/$DATASET/results.tsv" | while IFS=$'\t' read -r exp desc bpb rest; do
        printf "  %-6s %8s  %s\n" "$exp" "$bpb" "$desc"
    done
fi

echo
echo "  Next steps:"
echo "    - Review: cat results/$DATASET/results.tsv"
echo "    - Resume: bash platforms/runpod/scripts/launch.sh --max=50"
echo "    - Stop pod to save credits!"
echo
