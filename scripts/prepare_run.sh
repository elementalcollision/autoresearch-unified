#!/bin/bash
# =============================================================================
# Prepare a clean autoresearch run in WSL
#
# Resets the workspace, archives old results, and gets ready for a fresh run.
#
# Usage (from Windows PowerShell):
#   wsl -d Ubuntu-22.04 bash /mnt/c/Users/patri/autoresearch-unified/scripts/prepare_run.sh --run 3 --model kimik25
#
# Usage (from WSL):
#   bash /mnt/c/Users/patri/autoresearch-unified/scripts/prepare_run.sh --run 3 --model kimik25
# =============================================================================
set -euo pipefail

AUTORESEARCH_DIR="/home/pat/autoresearch"
RUN=0
MODEL=""
TAG=""
MAX=100

for arg in "$@"; do
    case $arg in
        --run=*)    RUN="${arg#*=}" ;;
        --model=*)  MODEL="${arg#*=}" ;;
        --tag=*)    TAG="${arg#*=}" ;;
        --max=*)    MAX="${arg#*=}" ;;
        --help)
            echo "Usage: bash prepare_run.sh --run=N --model=SLUG"
            echo "  --run=N       Run number (1, 2, 3)"
            echo "  --model=SLUG  Model slug (kimik25, qwen35-397b, sonnet46, gpt41)"
            echo "  --tag=TAG     Override run tag (default: controlled-SLUG-rN)"
            echo "  --max=N       Max experiments (default: 100)"
            exit 0 ;;
        *) echo "Unknown: $arg (try --help)"; exit 1 ;;
    esac
done

if [ "$RUN" -eq 0 ] || [ -z "$MODEL" ]; then
    echo "ERROR: --run and --model are required"
    echo "Example: bash prepare_run.sh --run=3 --model=kimik25"
    exit 1
fi

[ -z "$TAG" ] && TAG="controlled-${MODEL}-r${RUN}"

echo "============================================================"
echo "  Preparing Run: $MODEL R$RUN"
echo "  Tag: $TAG"
echo "  Max experiments: $MAX"
echo "  Workspace: $AUTORESEARCH_DIR"
echo "============================================================"
echo

cd "$AUTORESEARCH_DIR"

# Step 1: Archive existing results
echo "[1/5] Archive old results..."
ARCHIVE_DIR="$AUTORESEARCH_DIR/archives"
mkdir -p "$ARCHIVE_DIR"

if [ -f agent_results_pubmed.tsv ]; then
    ARCHIVE_NAME="agent_results_pubmed_$(date +%Y%m%d_%H%M%S).tsv"
    cp agent_results_pubmed.tsv "$ARCHIVE_DIR/$ARCHIVE_NAME"
    echo "  Archived: $ARCHIVE_DIR/$ARCHIVE_NAME"
    rm -f agent_results_pubmed.tsv
    echo "  Removed old results file"
else
    echo "  No existing results to archive"
fi

if [ -f agent_pubmed.log ]; then
    LOG_ARCHIVE="agent_pubmed_$(date +%Y%m%d_%H%M%S).log"
    cp agent_pubmed.log "$ARCHIVE_DIR/$LOG_ARCHIVE"
    echo "  Archived log: $ARCHIVE_DIR/$LOG_ARCHIVE"
    rm -f agent_pubmed.log
else
    echo "  No existing log to archive"
fi
echo

# Step 2: Reset training script to baseline
echo "[2/5] Reset training script to baseline..."
if git diff --quiet -- train.py 2>/dev/null; then
    echo "  train.py is clean (no local changes)"
else
    echo "  WARNING: train.py has local changes — resetting to master"
    git checkout master -- train.py 2>/dev/null || git checkout main -- train.py
fi
echo

# Step 3: Remove stale PID lock
echo "[3/5] Clean up stale locks..."
rm -f .suite.pid
echo "  Removed .suite.pid"
echo

# Step 4: Verify environment
echo "[4/5] Verify environment..."
if [ -d .venv ]; then
    echo "  Python venv: found"
else
    echo "  WARNING: No .venv found — you may need to run setup first"
fi

if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    echo "  OPENROUTER_API_KEY: set (${#OPENROUTER_API_KEY} chars)"
elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  ANTHROPIC_API_KEY: set (${#ANTHROPIC_API_KEY} chars)"
else
    echo "  WARNING: No API key set! Export OPENROUTER_API_KEY or ANTHROPIC_API_KEY"
fi
echo

# Step 5: Print the run command
echo "[5/5] Ready to run!"
echo
echo "============================================================"
echo "  Workspace is clean. Run this command to start:"
echo
echo "  cd $AUTORESEARCH_DIR"

# Detect which API to use
if [ -n "${OPENROUTER_API_KEY:-}" ]; then
    case "$MODEL" in
        kimik25)     MODEL_ID="moonshotai/kimi-k2.5" ;;
        qwen35-397b) MODEL_ID="qwen/qwen3.5-397b-a17b" ;;
        sonnet46)    MODEL_ID="anthropic/claude-sonnet-4.6" ;;
        gpt41)       MODEL_ID="openai/gpt-4.1" ;;
        *)           MODEL_ID="$MODEL" ;;
    esac
    echo "  python -m tui.headless --tag $TAG --max $MAX --results agent_results_pubmed.tsv --dataset pubmed --model $MODEL_ID 2>&1 | tee agent_pubmed.log"
else
    echo "  python -m tui.headless --tag $TAG --max $MAX --results agent_results_pubmed.tsv --dataset pubmed 2>&1 | tee agent_pubmed.log"
fi

echo
echo "  When done, publish with:"
echo "  cd /mnt/c/Users/patri/autoresearch-unified"
echo "  python scripts/publish_run.py --model=$MODEL --run=$RUN"
echo "============================================================"
