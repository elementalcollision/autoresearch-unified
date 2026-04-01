#!/usr/bin/env bash
# digitalocean-h200.sh — Repeatable provisioning and benchmarking for NVIDIA H200 on DigitalOcean
#
# Usage:
#   ./scripts/cloud/digitalocean-h200.sh provision              # Create GPU droplet
#   ./scripts/cloud/digitalocean-h200.sh setup <ANTHROPIC_KEY>   # Install deps on remote
#   ./scripts/cloud/digitalocean-h200.sh run [N]                 # Run benchmark (N runs, default 3)
#   ./scripts/cloud/digitalocean-h200.sh log [N]                 # Show last N log lines
#   ./scripts/cloud/digitalocean-h200.sh progress                # Check experiment counts
#   ./scripts/cloud/digitalocean-h200.sh collect                 # Pull results to local
#   ./scripts/cloud/digitalocean-h200.sh teardown                # Destroy droplet
#   ./scripts/cloud/digitalocean-h200.sh status                  # Show droplet status
#   ./scripts/cloud/digitalocean-h200.sh ssh                     # Open SSH session
#
# Prerequisites:
#   brew install doctl && doctl auth init

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
STATE_FILE="${PROJECT_ROOT}/tools/digitalocean-h200-state.json"

SIZE="gpu-h200x1-141gb"
IMAGE="gpu-h100x1-base"           # NVIDIA AI/ML Ready base image (works for H200)
REGION="tor1"                      # Toronto — primary GPU region
LABEL="autoresearch-h200"
REPO_URL="https://github.com/elementalcollision/autoresearch-unified.git"
CONTRIBUTOR="elementalcollision"
GPU_SLUG="h200"
MODEL="claude-haiku-4-5-20251001"
MODEL_SLUG="haiku45"
MAX_EXPERIMENTS=80
DATASETS="climbmix pubmed"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

require_doctl() {
    command -v doctl &>/dev/null || die "doctl not found — install: brew install doctl && doctl auth init"
}

require_state() {
    [[ -f "$STATE_FILE" ]] || die "No state file at $STATE_FILE — run: $0 provision"
}

get_state() {
    python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('$1', ''))"
}

save_state() {
    python3 -c "
import json, os
path = '$STATE_FILE'
d = json.load(open(path)) if os.path.exists(path) else {}
d['$1'] = '$2'
json.dump(d, open(path, 'w'), indent=2)
"
}

get_ip() {
    require_state
    local ip
    ip="$(get_state ip)"
    [[ -n "$ip" ]] || die "No IP in state file — droplet may still be provisioning"
    echo "$ip"
}

remote() {
    local ip
    ip="$(get_ip)"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@${ip}" "$@"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_provision() {
    require_doctl

    # Verify the H200 size is available
    info "Checking H200 availability..."
    local size_info
    size_info=$(doctl compute size list --no-header 2>&1 | grep "$SIZE" || true)
    if [[ -z "$size_info" ]]; then
        info "Available GPU sizes:"
        doctl compute size list --no-header 2>&1 | grep "gpu-" || true
        die "Size $SIZE not found. Check with: doctl compute size list | grep gpu"
    fi
    info "Found: $size_info"

    # Pick SSH key
    info "Checking SSH keys..."
    local ssh_key_id
    ssh_key_id=$(doctl compute ssh-key list --no-header --format ID 2>&1 | head -1)
    [[ -n "$ssh_key_id" ]] || die "No SSH keys found. Add one: doctl compute ssh-key create"
    info "SSH key ID: $ssh_key_id"

    # Check region availability for this size
    info "Using region: $REGION"

    # Create the droplet
    info "Creating GPU droplet ($SIZE in $REGION)..."
    local result
    result=$(doctl compute droplet create "$LABEL" \
        --size "$SIZE" \
        --image "$IMAGE" \
        --region "$REGION" \
        --ssh-keys "$ssh_key_id" \
        --wait \
        --format ID,PublicIPv4,Status \
        --no-header 2>&1)

    if [[ $? -ne 0 ]]; then
        die "Failed to create droplet: $result"
    fi

    local droplet_id ip status
    droplet_id=$(echo "$result" | awk '{print $1}')
    ip=$(echo "$result" | awk '{print $2}')
    status=$(echo "$result" | awk '{print $3}')

    [[ -n "$droplet_id" ]] || die "Failed to parse droplet ID from: $result"

    info "Droplet created: $droplet_id (status: $status)"
    save_state "id" "$droplet_id"
    save_state "region" "$REGION"

    # doctl --wait should give us the IP, but poll if needed
    if [[ -z "$ip" || "$ip" == "0" ]]; then
        info "Waiting for IP assignment..."
        for i in $(seq 1 30); do
            sleep 10
            ip=$(doctl compute droplet get "$droplet_id" --format PublicIPv4 --no-header 2>/dev/null || echo "")
            info "  [$i/30] ip=$ip"
            if [[ -n "$ip" && "$ip" != "0" ]]; then
                break
            fi
        done
    fi

    [[ -n "$ip" && "$ip" != "0" ]] || die "Timed out waiting for IP"

    save_state "ip" "$ip"
    save_state "status" "active"
    info "Droplet ready at $ip"
    info ""
    info "Next step:"
    info "  $0 setup <ANTHROPIC_API_KEY>"
}

cmd_setup() {
    local api_key="${1:-}"
    [[ -n "$api_key" ]] || die "Usage: $0 setup <ANTHROPIC_API_KEY>"

    local ip
    ip="$(get_ip)"
    info "Setting up remote environment on $ip..."

    # Wait for SSH to become available (droplet may still be booting)
    info "Waiting for SSH..."
    for i in $(seq 1 30); do
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
            "root@${ip}" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
            info "SSH ready"
            break
        fi
        info "  [$i/30] waiting..."
        sleep 10
    done

    # Transfer setup commands via SSH
    remote bash -s <<SETUP_EOF
set -euo pipefail

echo "==> Checking NVIDIA driver..."
nvidia-smi || { echo "ERROR: nvidia-smi not found"; exit 1; }
echo ""

echo "==> Installing uv..."
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.local/bin:\$PATH"
fi
uv --version

echo "==> Cloning repository..."
if [[ ! -d autoresearch-unified ]]; then
    git clone ${REPO_URL}
fi
cd autoresearch-unified

echo "==> Installing dependencies..."
uv venv
uv pip install -e ".[all-cuda]"

echo "==> Setting API key..."
echo "ANTHROPIC_API_KEY=${api_key}" > .env

echo "==> Verifying GPU detection..."
uv run python -c "
from backends import get_hardware_info
info = get_hardware_info()
print(f'GPU:       {info.get(\"chip_name\", \"unknown\")}')
print(f'Memory:    {info.get(\"memory_gb\", 0):.0f} GB')
print(f'Tier:      {info.get(\"chip_tier\", \"unknown\")}')
print(f'Cores/SMs: {info.get(\"gpu_cores\", 0)}')
assert info.get('chip_tier') == 'datacenter', f'Expected datacenter tier, got {info.get(\"chip_tier\")}'
print('GPU detection OK')
"

echo "==> Verifying nvidia-ml-py power metrics..."
uv run python -c "
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
print(f'Power draw: {power_mw / 1000:.1f}W')
assert power_mw > 0, 'nvidia-ml-py returned 0 power — check GPU'
print('nvidia-ml-py power metrics OK')
"

echo "==> Verifying PowerMonitor integration..."
uv run python -c "
from backends.power import PowerMonitor
import time
m = PowerMonitor('cuda')
assert m._sampler is not None, 'PowerMonitor has no sampler — nvidia-ml-py not working'
m.start()
time.sleep(2)
avg_watts, total_joules = m.stop(2.0)
print(f'PowerMonitor: avg={avg_watts:.1f}W, joules={total_joules:.1f}J')
assert avg_watts > 0, 'PowerMonitor returned 0W'
print('PowerMonitor integration OK')
"

echo ""
echo "=== Setup complete ==="
SETUP_EOF

    save_state "status" "setup_complete"
    info "Remote setup complete"
    info ""
    info "Next step:"
    info "  $0 run 3"
}

cmd_run() {
    local num_runs="${1:-3}"
    local ip
    ip="$(get_ip)"

    info "Starting $num_runs benchmark runs on $ip (tmux session, survives disconnects)..."

    # Upload the runner script to the remote
    remote bash -c "cat > /root/autoresearch-unified/run_benchmark.sh && chmod +x /root/autoresearch-unified/run_benchmark.sh" <<'RUNNER_EOF'
#!/usr/bin/env bash
# run_benchmark.sh — Runs inside tmux on the remote. Logs everything.
# Supports resume: skips run+dataset combos that already have results.
set -euo pipefail

NUM_RUNS="${1:-3}"
DATASETS="${2:-climbmix pubmed}"
MODEL="claude-haiku-4-5-20251001"
MODEL_SLUG="haiku45"
MAX_EXPERIMENTS=80
CONTRIBUTOR="elementalcollision"
GPU_SLUG="h200"

export PATH="$HOME/.local/bin:$PATH"
cd /root/autoresearch-unified

# Source API key
set -a; source .env; set +a

LOGFILE="/root/autoresearch-unified/logs/benchmark.log"
mkdir -p "$(dirname "$LOGFILE")"

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$LOGFILE"
}

log "=== Benchmark started: $NUM_RUNS runs, datasets: $DATASETS ==="
log "Model: $MODEL | Max experiments: $MAX_EXPERIMENTS | GPU: $GPU_SLUG"

for run in $(seq 1 "$NUM_RUNS"); do
    for dataset in $DATASETS; do
        dst_dir="data/results/${dataset}/${CONTRIBUTOR}-${GPU_SLUG}"
        dst_file="${dst_dir}/results_${MODEL_SLUG}_r${run}.tsv"

        # Resume support: skip if this run already has results
        if [[ -f "$dst_file" ]]; then
            log "SKIP: $dst_file already exists (run $run, $dataset)"
            continue
        fi

        log "START: run $run/$NUM_RUNS, dataset=$dataset"

        # Run the suite for this dataset
        if uv run run_suite.py \
            --model "$MODEL" \
            --max-experiments "$MAX_EXPERIMENTS" \
            --dataset "$dataset" \
            >> "$LOGFILE" 2>&1; then
            log "DONE: run_suite.py completed for $dataset (run $run)"
        else
            log "ERROR: run_suite.py failed for $dataset (run $run), exit code $?"
            log "  Check $LOGFILE for details. Continuing to next dataset..."
            continue
        fi

        # Find the results file (model slug format varies)
        src=""
        for candidate in \
            "results/haiku-4-5-20251001/${dataset}/results.tsv" \
            "results/${MODEL_SLUG}/${dataset}/results.tsv"; do
            if [[ -f "$candidate" ]]; then
                src="$candidate"
                break
            fi
        done

        if [[ -z "$src" ]]; then
            log "ERROR: No results file found for $dataset (run $run)"
            continue
        fi

        # Copy to leaderboard directory
        mkdir -p "$dst_dir"
        cp "$src" "$dst_file"
        log "SAVED: $dst_file ($(wc -l < "$dst_file") lines)"

        # Clear working results for next run
        rm -f "$src"
    done
done

log "=== Benchmark complete ==="
log "Results summary:"
find data/results -name "results_${MODEL_SLUG}_*.tsv" -type f | while read -r f; do
    log "  $f ($(wc -l < "$f") lines)"
done
RUNNER_EOF

    # Ensure tmux is installed, then launch
    remote bash -s -- "$num_runs" "$DATASETS" <<'LAUNCH_EOF'
set -euo pipefail
apt-get install -y tmux > /dev/null 2>&1 || true

NUM_RUNS="$1"
DATASETS="$2"

# Kill existing session if any
tmux kill-session -t benchmark 2>/dev/null || true

# Start new detached tmux session
tmux new-session -d -s benchmark \
    "bash /root/autoresearch-unified/run_benchmark.sh '$NUM_RUNS' '$DATASETS'; echo 'DONE — press enter to close'; read"

echo "tmux session 'benchmark' started"
LAUNCH_EOF

    save_state "status" "running"
    info "Benchmark launched in tmux session 'benchmark'"
    info ""
    info "Monitor remotely:"
    info "  $0 ssh                                    # then: tmux attach -t benchmark"
    info "  $0 log                                    # tail the log file"
    info "  $0 progress                               # check experiment counts"
}

cmd_log() {
    local lines="${1:-50}"
    info "Last $lines lines of benchmark log:"
    remote tail -n "$lines" /root/autoresearch-unified/logs/benchmark.log 2>/dev/null || \
        echo "No log file yet."
}

cmd_progress() {
    info "Checking progress..."
    remote bash -s <<'PROGRESS_EOF'
cd /root/autoresearch-unified 2>/dev/null || exit 1

echo "=== Leaderboard results ==="
find data/results -name "results_haiku45_*.tsv" -type f 2>/dev/null | while read -r f; do
    lines=$(( $(wc -l < "$f") - 1 ))  # subtract header
    echo "  $f — $lines experiments"
done

echo ""
echo "=== Working results (in progress) ==="
find results -name "results.tsv" -type f 2>/dev/null | while read -r f; do
    lines=$(( $(wc -l < "$f") - 1 ))
    echo "  $f — $lines experiments"
done

echo ""
echo "=== tmux session ==="
tmux has-session -t benchmark 2>/dev/null && echo "  benchmark: RUNNING" || echo "  benchmark: NOT RUNNING"

echo ""
echo "=== Last 5 log entries ==="
tail -5 /root/autoresearch-unified/logs/benchmark.log 2>/dev/null || echo "  (no log yet)"
PROGRESS_EOF
}

cmd_collect() {
    local ip
    ip="$(get_ip)"

    info "Collecting results from $ip..."

    local local_results="${PROJECT_ROOT}/data/results"

    for dataset in $DATASETS; do
        local remote_dir="autoresearch-unified/data/results/${dataset}/${CONTRIBUTOR}-${GPU_SLUG}"
        local local_dir="${local_results}/${dataset}/${CONTRIBUTOR}-${GPU_SLUG}"
        mkdir -p "$local_dir"

        info "  Syncing ${dataset}..."
        rsync -avz \
            -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
            "root@${ip}:${remote_dir}/" \
            "${local_dir}/"
    done

    info "Rebuilding leaderboard..."
    cd "$PROJECT_ROOT"
    python3 scripts/build_benchmark.py

    save_state "status" "collected"
    info "Results collected and leaderboard rebuilt"
    info "Check: docs/benchmark/data.json"
}

cmd_teardown() {
    require_doctl
    require_state

    local droplet_id
    droplet_id="$(get_state id)"
    [[ -n "$droplet_id" ]] || die "No droplet ID in state file"

    info "Destroying droplet $droplet_id..."
    doctl compute droplet delete "$droplet_id" --force

    save_state "status" "destroyed"
    info "Droplet destroyed. Billing stopped."
}

cmd_status() {
    require_doctl
    require_state

    local droplet_id
    droplet_id="$(get_state id)"
    [[ -n "$droplet_id" ]] || die "No droplet ID in state file"

    doctl compute droplet get "$droplet_id" --format ID,Name,PublicIPv4,Status,Size,Region,Created
}

cmd_ssh() {
    local ip
    ip="$(get_ip)"
    exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "root@${ip}"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

case "${1:-}" in
    provision)  cmd_provision ;;
    setup)      shift; cmd_setup "$@" ;;
    run)        shift; cmd_run "$@" ;;
    log)        shift; cmd_log "$@" ;;
    progress)   cmd_progress ;;
    collect)    cmd_collect ;;
    teardown)   cmd_teardown ;;
    status)     cmd_status ;;
    ssh)        cmd_ssh ;;
    *)
        echo "Usage: $0 {provision|setup|run|log|progress|collect|teardown|status|ssh}"
        echo ""
        echo "  provision              Create H200 GPU droplet on DigitalOcean"
        echo "  setup <API_KEY>        Install deps + verify GPU on remote"
        echo "  run [N]                Run N benchmark rounds (default: 3)"
        echo "  log [N]                Show last N lines of benchmark log (default: 50)"
        echo "  progress               Check experiment counts and tmux status"
        echo "  collect                Pull results and rebuild leaderboard"
        echo "  teardown               Destroy droplet (stops billing)"
        echo "  status                 Show droplet status"
        echo "  ssh                    Open SSH session to droplet"
        echo ""
        echo "  Cost: \$3.44/hr (~\$145 for full 3-run benchmark)"
        exit 1
        ;;
esac
