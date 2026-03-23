#!/bin/bash
# =============================================================================
# stop.sh — Graceful shutdown for RunPod experiments
#
# Run this BEFORE stopping or terminating a RunPod pod.
# It ensures all experiment data is saved and synced.
#
# What it does:
#   1. Sends SIGTERM to the headless runner (triggers graceful cleanup)
#   2. Waits for the runner to finish current experiment
#   3. Runs a final results sync to volume + GitHub
#   4. Prints summary of what was saved
#
# Usage:
#   bash platforms/runpod/scripts/stop.sh
# =============================================================================
set -euo pipefail

WORKSPACE="${AUTORESEARCH_WORKSPACE:-/workspace/autoresearch-unified}"
cd "$WORKSPACE"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo
echo -e "${BOLD}  autoresearch — Graceful Shutdown${NC}"
echo

# ── 1. Stop the headless runner ────────────────────────────
if [ -f .suite.pid ]; then
    PID=$(cat .suite.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "  ${BLUE}Sending SIGTERM to runner (PID $PID)...${NC}"
        kill -TERM "$PID"
        echo "  Waiting for current experiment to finish (up to 6 min)..."
        for i in $(seq 1 72); do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "  ${GREEN}Runner stopped cleanly.${NC}"
                break
            fi
            sleep 5
        done
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "  ${YELLOW}Runner still alive after 6 min — sending SIGKILL${NC}"
            kill -9 "$PID" 2>/dev/null || true
        fi
    else
        echo "  Runner not running (stale PID file)."
    fi
else
    echo "  No PID file found — runner may not be active."
fi

# ── 2. Final sync ─────────────────────────────────────────
SYNC_SCRIPT="$WORKSPACE/platforms/runpod/scripts/sync_results.sh"
if [ -f "$SYNC_SCRIPT" ]; then
    echo
    echo -e "  ${BLUE}Running final results sync...${NC}"
    bash "$SYNC_SCRIPT"
    echo -e "  ${GREEN}Sync complete.${NC}"
else
    echo -e "  ${YELLOW}Sync script not found — manual backup needed.${NC}"
fi

# ── 3. Summary ─────────────────────────────────────────────
echo
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}  Shutdown Summary${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# Show results summary
for f in results_*.tsv results/*/results.tsv; do
    if [ -f "$f" ]; then
        TOTAL=$(($(wc -l < "$f") - 1))
        BEST=$(tail -n +2 "$f" | awk -F'\t' '$3 > 0 {print $3}' | sort -n | head -1)
        echo -e "  ${GREEN}Results: $f ($TOTAL experiments, best val_bpb=$BEST)${NC}"
    fi
done

# Show where data is saved
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo
echo "  Data saved to:"
echo "    GitHub: origin/$BRANCH"
if [ -d /runpod-volume/autoresearch-backup ]; then
    echo "    Volume: /runpod-volume/autoresearch-backup/"
fi
echo "    Local:  $WORKSPACE"
echo
echo -e "  ${GREEN}Safe to stop or terminate the pod.${NC}"
echo
