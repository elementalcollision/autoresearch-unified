#!/bin/bash
# =============================================================================
# Controlled n=3 Run Script
#
# Runs the headless experiment orchestrator N times with a fresh baseline
# reset between each run. Each run produces a separate results TSV file.
#
# Usage:
#   export OPENROUTER_API_KEY="sk-or-..."
#   bash platforms/runpod/scripts/run_controlled.sh \
#       --model="qwen/qwen3.5-397b-a17b" --slug=qwen35-397b --max=100 --n=3
#
# Environment:
#   OPENROUTER_API_KEY / ANTHROPIC_API_KEY — at least one required
#   Must be run from the autoresearch-unified workspace root.
# =============================================================================
set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────
MODEL="${OPENROUTER_MODEL:-anthropic/claude-sonnet-4.6}"
SLUG="controlled"
MAX=100
N_RUNS=3
DATASET="pubmed"
TAG="controlled"

# ── Parse CLI args ──────────────────────────────────────────────
for arg in "$@"; do
    case $arg in
        --model=*)   MODEL="${arg#*=}" ;;
        --slug=*)    SLUG="${arg#*=}" ;;
        --max=*)     MAX="${arg#*=}" ;;
        --n=*)       N_RUNS="${arg#*=}" ;;
        --dataset=*) DATASET="${arg#*=}" ;;
        --tag=*)     TAG="${arg#*=}" ;;
        --help)
            echo "Usage: bash run_controlled.sh [OPTIONS]"
            echo "  --model=MODEL    OpenRouter/API model ID (default: \$OPENROUTER_MODEL)"
            echo "  --slug=SLUG      Short model slug for filenames (default: controlled)"
            echo "  --max=N          Max experiments per run (default: 100)"
            echo "  --n=N            Number of independent runs (default: 3)"
            echo "  --dataset=NAME   Dataset name (default: pubmed)"
            echo "  --tag=TAG        Run tag (default: controlled)"
            exit 0 ;;
        *) echo "Unknown: $arg (try --help)"; exit 1 ;;
    esac
done

# ── Detect backend for training script reset ────────────────────
BACKEND="${AUTORESEARCH_BACKEND:-cuda}"
case $BACKEND in
    cuda)        TRAIN_SCRIPT="platforms/cuda/train_cuda.py" ;;
    rocm|rocm7)  TRAIN_SCRIPT="platforms/rocm/train_rocm.py" ;;
    *)           TRAIN_SCRIPT="platforms/cuda/train_cuda.py" ;;
esac

# ── Validate ────────────────────────────────────────────────────
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY" >&2
    exit 1
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    exit 1
fi

SYNC_SCRIPT="platforms/runpod/scripts/sync_results.sh"

echo "============================================================"
echo "  Controlled Run: n=$N_RUNS x $MAX experiments"
echo "  Model: $MODEL"
echo "  Slug: $SLUG"
echo "  Dataset: $DATASET"
echo "  Training script: $TRAIN_SCRIPT"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo

for RUN in $(seq 1 "$N_RUNS"); do
    echo "━━━ Run $RUN of $N_RUNS ━━━"
    RUN_START=$(date +%s)

    # Reset training script to clean baseline (no HP modifications)
    echo "  Resetting $TRAIN_SCRIPT to baseline..."
    git checkout main -- "$TRAIN_SCRIPT"

    # Remove stale PID lock from previous run
    rm -f .suite.pid

    RESULTS_FILE="results_${SLUG}_r${RUN}.tsv"
    echo "  Results: $RESULTS_FILE"
    echo "  Starting at: $(date '+%H:%M:%S')"
    echo

    python -m tui.headless \
        --tag "${TAG}" \
        --max "$MAX" \
        --results "$RESULTS_FILE" \
        --dataset "$DATASET" \
        --model "$MODEL" \
        2>&1 | tee "/tmp/agent_run_r${RUN}.log"

    # Sync results after each run
    if [ -f "$SYNC_SCRIPT" ]; then
        bash "$SYNC_SCRIPT" 2>/dev/null || true
    fi

    RUN_ELAPSED=$(( $(date +%s) - RUN_START ))
    echo
    echo "  Run $RUN complete: $((RUN_ELAPSED/60))m $((RUN_ELAPSED%60))s"

    if [ -f "$RESULTS_FILE" ]; then
        TOTAL=$(($(wc -l < "$RESULTS_FILE") - 1))
        KEPT=$(tail -n +2 "$RESULTS_FILE" | awk -F'\t' '$4 == "keep" {c++} END {print c+0}')
        echo "  Experiments: $TOTAL | Kept: $KEPT"
    fi
    echo
done

echo "============================================================"
echo "  All $N_RUNS runs complete: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Results files:"
for RUN in $(seq 1 "$N_RUNS"); do
    FILE="results_${SLUG}_r${RUN}.tsv"
    [ -f "$FILE" ] && echo "    $FILE ($(( $(wc -l < "$FILE") - 1 )) experiments)"
done
echo "============================================================"
