#!/bin/bash
# =============================================================================
# autoresearch-gaudi — Launch Day Script
#
# Run this ON the Gaudi 3 instance. It does everything:
#   1. Validate environment (hl-smi, Docker, Habana runtime)
#   2. Clone repo + build Docker image
#   3. Verify HPU inside container
#   4. Download + prepare ClimbMix data
#   5. Run a baseline training (5 minutes)
#   6. Launch autonomous agent (ClimbMix benchmark)
#
# Prerequisites:
#   - IBM Cloud Gaudi 3 instance (gx3d-160x1792x8gaudi3) is running
#   - You've SSH'd into the instance
#   - ANTHROPIC_API_KEY is set (or will be prompted)
#
# Usage:
#   # Full automated run:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash launch.sh
#
#   # Step-by-step (stop after each phase):
#   bash launch.sh --step
#
#   # Verify only (no training):
#   bash launch.sh --verify-only
#
#   # Skip to agent (data already prepared):
#   bash launch.sh --agent-only
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
REPO_URL="https://github.com/elementalcollision/autoresearch-gaudi.git"
WORK_DIR="/workspace/autoresearch"
TAG=$(date +%b%d | tr '[:upper:]' '[:lower:]')
MAX_EXPERIMENTS=${MAX_EXPERIMENTS:-30}
NUM_SHARDS=${NUM_SHARDS:-10}
STEP_MODE=false
VERIFY_ONLY=false
AGENT_ONLY=false
SKIP_BUILD=false

# Parse args
for arg in "$@"; do
    case $arg in
        --step)       STEP_MODE=true ;;
        --verify-only) VERIFY_ONLY=true ;;
        --agent-only)  AGENT_ONLY=true ;;
        --skip-build)  SKIP_BUILD=true ;;
        --max=*)       MAX_EXPERIMENTS="${arg#*=}" ;;
        --shards=*)    NUM_SHARDS="${arg#*=}" ;;
        --tag=*)       TAG="${arg#*=}" ;;
        *)             echo "Unknown arg: $arg"; exit 1 ;;
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
    echo -e "  ${YELLOW}⏱  Elapsed: ${min}m ${sec}s${NC}"
}

step_pause() {
    if $STEP_MODE; then
        elapsed
        echo -e "\n  ${YELLOW}--step mode: Press Enter to continue, Ctrl+C to stop${NC}"
        read -r
    fi
}

fail_exit() {
    echo -e "\n  ${RED}FATAL: $1${NC}"
    elapsed
    exit 1
}

# =============================================================================
# Phase 0: Pre-checks
# =============================================================================
phase "0" "Environment validation"

echo "  Host: $(hostname)"
echo "  OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
echo "  Kernel: $(uname -r)"
echo "  Date: $(date)"
echo "  Tag: $TAG"
echo "  Max experiments: $MAX_EXPERIMENTS"

# Check hl-smi
echo
echo "  Checking Gaudi 3 drivers..."
if command -v hl-smi &>/dev/null; then
    DEVICE_COUNT=$(hl-smi -Q name -f csv,noheader 2>/dev/null | wc -l)
    DEVICE_NAME=$(hl-smi -Q name -f csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')
    echo -e "  ${GREEN}hl-smi: ${DEVICE_COUNT} x ${DEVICE_NAME}${NC}"
    if [ "$DEVICE_COUNT" -lt 1 ]; then
        fail_exit "No Gaudi devices detected"
    fi
else
    fail_exit "hl-smi not found — habanalabs drivers not installed"
fi

# Check Docker
echo "  Checking Docker..."
if command -v docker &>/dev/null; then
    echo -e "  ${GREEN}Docker: $(docker --version | head -1)${NC}"
else
    fail_exit "Docker not installed"
fi

# Check Habana runtime
if docker info 2>/dev/null | grep -qi "habana"; then
    echo -e "  ${GREEN}Habana container runtime: configured${NC}"
else
    echo -e "  ${YELLOW}Habana runtime not in docker info — will try anyway${NC}"
fi

# Check ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo
    echo -e "  ${YELLOW}ANTHROPIC_API_KEY not set.${NC}"
    echo -n "  Enter your Anthropic API key (or press Enter to skip agent): "
    read -r ANTHROPIC_API_KEY
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "  Proceeding without API key (baseline only, no agent)"
    else
        export ANTHROPIC_API_KEY
    fi
else
    echo -e "  ${GREEN}ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)${NC}"
fi

step_pause

# =============================================================================
# Phase 1: Clone & Build
# =============================================================================
if ! $AGENT_ONLY; then

phase "1" "Clone repository & build Docker image"

if [ -d "$WORK_DIR/.git" ]; then
    echo "  Repo already cloned at $WORK_DIR, pulling latest..."
    cd "$WORK_DIR"
    git pull --ff-only 2>/dev/null || echo "  (pull skipped — may be on experiment branch)"
else
    echo "  Cloning $REPO_URL..."
    git clone "$REPO_URL" "$WORK_DIR"
    cd "$WORK_DIR"
fi

echo "  Current commit: $(git log --oneline -1)"

if $SKIP_BUILD; then
    echo "  --skip-build: skipping Docker build"
else
    echo
    echo "  Building Docker image (this takes 2-5 minutes)..."
    BUILD_START=$(date +%s)
    docker compose build 2>&1 | tail -5
    BUILD_END=$(date +%s)
    echo -e "  ${GREEN}Build complete in $((BUILD_END - BUILD_START))s${NC}"
fi

step_pause

# =============================================================================
# Phase 2: HPU Verification
# =============================================================================
phase "2" "HPU verification (inside container)"

echo "  Running verify_hpu.py..."
if docker compose run --rm verify; then
    echo -e "\n  ${GREEN}HPU verification passed${NC}"
else
    fail_exit "HPU verification failed — check drivers and runtime"
fi

if $VERIFY_ONLY; then
    elapsed
    echo -e "\n  ${GREEN}--verify-only: stopping here. Environment is ready.${NC}"
    exit 0
fi

step_pause

# =============================================================================
# Phase 3: Download & Prepare ClimbMix Data
# =============================================================================
phase "3" "Prepare ClimbMix data (${NUM_SHARDS} shards)"

echo "  Downloading from HuggingFace + training tokenizer..."
PREP_START=$(date +%s)
docker compose run --rm prepare python -u prepare.py --num-shards="$NUM_SHARDS"
PREP_END=$(date +%s)
echo -e "  ${GREEN}Data prep complete in $((PREP_END - PREP_START))s${NC}"

# Verify data exists
echo "  Verifying data..."
docker compose run --rm prepare python -c "
import os
from pathlib import Path
data_dir = Path.home() / '.cache' / 'autoresearch' / 'data'
tok_dir = Path.home() / '.cache' / 'autoresearch' / 'tokenizer'
shards = sorted(data_dir.glob('shard_*.parquet'))
print(f'  Shards: {len(shards)}')
print(f'  Tokenizer: {\"OK\" if (tok_dir / \"tokenizer.bin\").exists() else \"MISSING\"}')
val = data_dir / 'shard_06542.parquet'
print(f'  Validation shard: {\"OK\" if val.exists() else \"MISSING\"}')
if not shards:
    raise SystemExit('  ERROR: No shards found')
if not val.exists():
    raise SystemExit('  ERROR: Validation shard missing')
" || fail_exit "Data verification failed"

step_pause

# =============================================================================
# Phase 4: Baseline Training Run
# =============================================================================
phase "4" "Baseline training (single HPU, ~5 min)"

echo "  Running train_gaudi.py..."
echo "  (first run includes torch.compile warmup — expect slow start)"
TRAIN_START=$(date +%s)
docker compose run --rm train 2>&1 | tee /tmp/baseline_run.log | tail -30
TRAIN_END=$(date +%s)
echo -e "  ${GREEN}Training complete in $((TRAIN_END - TRAIN_START))s${NC}"

# Extract key metrics from baseline
echo
echo "  Baseline results:"
grep -E "^(val_bpb|training_seconds|peak_vram_mb|mfu_percent|tok_sec|num_steps|depth|chip):" /tmp/baseline_run.log 2>/dev/null | sed 's/^/    /' || echo "    (could not parse — check /tmp/baseline_run.log)"

step_pause

fi  # end of !AGENT_ONLY block

# =============================================================================
# Phase 5: Autonomous Agent (ClimbMix Benchmark)
# =============================================================================
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    phase "5" "Autonomous agent (SKIPPED — no API key)"
    echo -e "  ${YELLOW}Set ANTHROPIC_API_KEY and run: bash launch.sh --agent-only${NC}"
else
    phase "5" "Autonomous agent — ClimbMix benchmark"

    cd "$WORK_DIR"

    echo "  Dataset: climbmix"
    echo "  Tag: $TAG"
    echo "  Max experiments: $MAX_EXPERIMENTS"
    echo "  Results: results/climbmix/results.tsv"
    echo

    # Ensure results directory exists
    mkdir -p results/climbmix

    echo "  Starting headless agent..."
    echo "  (Ctrl+C to stop gracefully)"
    echo

    docker compose run --rm \
        -e ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
        agent \
        python -m tui.headless \
            --tag "$TAG" \
            --max "$MAX_EXPERIMENTS" \
            --results results/climbmix/results.tsv \
        2>&1 | tee /tmp/agent_run.log

    echo
    echo "  Agent run complete."
    echo "  Results: results/climbmix/results.tsv"
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

if [ -f "$WORK_DIR/results/climbmix/results.tsv" ]; then
    echo
    echo "  Results summary:"
    head -1 "$WORK_DIR/results/climbmix/results.tsv"
    tail -n +2 "$WORK_DIR/results/climbmix/results.tsv" | while IFS=$'\t' read -r exp desc bpb mem tok mfu steps status notes gpu; do
        printf "  %-6s %-8s %8s  %s\n" "$exp" "$status" "$bpb" "$desc"
    done
fi

echo
echo "  Next steps:"
echo "    - Review: cat results/climbmix/results.tsv"
echo "    - Resume: bash scripts/launch.sh --agent-only --max=50"
echo "    - Benchmark: python scripts/benchmark_hpu.py"
echo "    - Stop instance to save credits!"
echo
