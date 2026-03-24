#!/bin/bash
# =============================================================================
# sync_results.sh — Periodic results sync for RunPod experiments
#
# Ensures experiment data survives pod termination by:
#   1. Copying results + heartbeat to the persistent RunPod volume
#   2. Committing results to git and pushing to the experiment branch
#
# Designed to be run:
#   - Manually:   bash platforms/runpod/scripts/sync_results.sh
#   - Via cron:   */10 * * * * /workspace/autoresearch-unified/platforms/runpod/scripts/sync_results.sh
#   - At launch:  Automatically installed by launch.sh
#
# Data flow:
#   Container disk (ephemeral)
#     └─► RunPod volume /runpod-volume/autoresearch-backup/ (survives stop)
#     └─► GitHub experiment branch (survives terminate)
#
# =============================================================================
set -euo pipefail

# ── Resolve workspace ──────────────────────────────────────
WORKSPACE="${AUTORESEARCH_WORKSPACE:-/workspace/autoresearch-unified}"
VOLUME_BACKUP="${AUTORESEARCH_VOLUME_BACKUP:-/runpod-volume/autoresearch-backup}"
LOGFILE="${WORKSPACE}/sync.log"

cd "$WORKSPACE" || exit 1

timestamp() { date -u +"%Y-%m-%d %H:%M:%S UTC"; }

log() { echo "[$(timestamp)] $*" >> "$LOGFILE"; }

# ── Find results files ─────────────────────────────────────
# Collect all .tsv results files (named results_*.tsv or under results/)
RESULTS_FILES=()
for f in results_*.tsv results/*/results.tsv; do
    [ -f "$f" ] && RESULTS_FILES+=("$f")
done

if [ ${#RESULTS_FILES[@]} -eq 0 ]; then
    log "SKIP: No results files found"
    exit 0
fi

# ── 1. Volume backup ──────────────────────────────────────
if [ -d "$(dirname "$VOLUME_BACKUP")" ]; then
    mkdir -p "$VOLUME_BACKUP"
    for f in "${RESULTS_FILES[@]}"; do
        # Preserve directory structure on volume
        dest="$VOLUME_BACKUP/$f"
        mkdir -p "$(dirname "$dest")"
        cp "$f" "$dest"
    done
    # Heartbeat
    [ -f .runner_status.json ] && cp .runner_status.json "$VOLUME_BACKUP/"
    # Git bundle (full branch snapshot — portable backup)
    BRANCH=$(git branch --show-current 2>/dev/null)
    if [ -n "$BRANCH" ] && [ "$BRANCH" != "main" ]; then
        git bundle create "$VOLUME_BACKUP/experiment-branch.bundle" "$BRANCH" 2>/dev/null || true
    fi
    log "VOLUME: Backed up ${#RESULTS_FILES[@]} results file(s) to $VOLUME_BACKUP"
else
    log "VOLUME: No RunPod volume mounted — skipping volume backup"
fi

# ── 2. Git commit + push ──────────────────────────────────
BRANCH=$(git branch --show-current 2>/dev/null)
if [ -z "$BRANCH" ]; then
    log "GIT: Not on a branch — skipping"
    exit 0
fi

# Ensure new branches auto-track remote on push (prevents silent push failures)
git config push.autoSetupRemote true 2>/dev/null

# Stage results and heartbeat
git add "${RESULTS_FILES[@]}" 2>/dev/null || true
git add .runner_status.json 2>/dev/null || true

# Only commit if there are staged changes
if ! git diff --cached --quiet 2>/dev/null; then
    # Build a useful commit message from heartbeat
    if [ -f .runner_status.json ] && command -v python3 &>/dev/null; then
        SUMMARY=$(python3 -c "
import json, sys
try:
    h = json.load(open('.runner_status.json'))
    print(f\"exp{h.get('experiment',0)} | kept={h.get('kept',0)} disc={h.get('discarded',0)} crash={h.get('crashes',0)} | best={h.get('best_bpb','?')}\")
except: print('in progress')
" 2>/dev/null)
    else
        SUMMARY="in progress"
    fi

    git commit -m "Sync results ($(date -u +%H:%M) UTC) — $SUMMARY" --no-verify 2>/dev/null
    log "GIT: Committed results"
else
    log "GIT: No changes to commit"
fi

# Push (only if remote tracking is set up)
if git rev-parse --abbrev-ref "@{u}" &>/dev/null; then
    if git push 2>/dev/null; then
        log "GIT: Pushed to origin/$BRANCH"
    else
        log "GIT: Push failed (will retry next cycle)"
    fi
else
    log "GIT: No upstream configured — skipping push"
fi
