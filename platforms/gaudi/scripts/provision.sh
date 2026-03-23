#!/bin/bash
# =============================================================================
# Provision a Gaudi 3 instance on IBM Cloud
#
# Run this from your LOCAL machine (laptop). It creates the instance,
# attaches a floating IP, waits for SSH, and copies launch.sh over.
#
# Prerequisites:
#   - ibmcloud CLI installed and logged in
#   - VPC 'autoresearch-vpc' exists (created during setup)
#   - SSH key 'autoresearch-key' registered
#   - Subnet 'autoresearch-subnet' exists
#
# Usage:
#   bash scripts/provision.sh                # Create instance + wait
#   bash scripts/provision.sh --status       # Check existing instance
#   bash scripts/provision.sh --destroy      # Tear down to stop billing
#   bash scripts/provision.sh --ssh          # SSH into running instance
#   bash scripts/provision.sh --cost         # Estimate session cost
# =============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────
INSTANCE_NAME="autoresearch-gaudi3"
VPC_NAME="autoresearch-vpc"
SUBNET_NAME="autoresearch-subnet"
KEY_NAME="autoresearch-key"
PROFILE="gx3d-160x1792x8gaudi3"
ZONE="us-east-2"
# Ubuntu 24.04 — fetch dynamically below
IMAGE_PATTERN="ibm-ubuntu-24-04"
PROMO_CODE="GPU4YOU"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

# ── Helpers ──────────────────────────────────────────────────

get_instance_id() {
    ibmcloud is instances --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for i in data:
    if i['name'] == '$INSTANCE_NAME':
        print(i['id'])
        break
" 2>/dev/null
}

get_floating_ip() {
    ibmcloud is floating-ips --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for ip in data:
    if ip.get('target', {}).get('name', '').startswith('$INSTANCE_NAME'):
        print(ip['address'])
        break
    # Also check by name
    if ip.get('name', '') == '${INSTANCE_NAME}-ip':
        print(ip['address'])
        break
" 2>/dev/null
}

get_instance_status() {
    ibmcloud is instances --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for i in data:
    if i['name'] == '$INSTANCE_NAME':
        print(i['status'])
        break
" 2>/dev/null
}

# ── Commands ─────────────────────────────────────────────────

cmd_status() {
    echo -e "${BLUE}Instance Status${NC}"
    local id=$(get_instance_id)
    if [ -z "$id" ]; then
        echo "  No instance named '$INSTANCE_NAME' found."
        return
    fi
    local status=$(get_instance_status)
    local ip=$(get_floating_ip)
    echo "  Name: $INSTANCE_NAME"
    echo "  ID: $id"
    echo "  Status: $status"
    echo "  IP: ${ip:-none}"
    if [ -n "$ip" ]; then
        echo
        echo "  SSH: ssh -o StrictHostKeyChecking=no root@$ip"
    fi
}

cmd_cost() {
    echo -e "${BLUE}Cost Estimate${NC}"
    echo
    echo "  Profile: $PROFILE"
    echo "  List price: ~\$60/hr"
    echo "  With $PROMO_CODE: ~\$30/hr (50% off, valid through June 2026)"
    echo
    echo "  Budget scenarios (\$200 credit, with promo):"
    echo "    1 hour:   ~\$30  — verify + baseline + ~5 experiments"
    echo "    2 hours:  ~\$60  — baseline + ~15 experiments (recommended)"
    echo "    3 hours:  ~\$90  — baseline + ~25 experiments"
    echo "    6.6 hours: \$200 — full budget"
    echo
    echo -e "  ${YELLOW}IMPORTANT: Apply promo code '$PROMO_CODE' at checkout!${NC}"
    echo -e "  ${YELLOW}Remember to destroy the instance when done to stop billing.${NC}"
}

cmd_ssh() {
    local ip=$(get_floating_ip)
    if [ -z "$ip" ]; then
        echo -e "${RED}No floating IP found. Is the instance running?${NC}"
        exit 1
    fi
    echo -e "  Connecting to ${BOLD}$ip${NC}..."
    exec ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"$ip"
}

cmd_destroy() {
    local id=$(get_instance_id)
    if [ -z "$id" ]; then
        echo "  No instance named '$INSTANCE_NAME' found."
        return
    fi

    echo -e "${RED}This will DESTROY instance '$INSTANCE_NAME' and stop all billing.${NC}"
    echo -n "  Type 'yes' to confirm: "
    read -r confirm
    if [ "$confirm" != "yes" ]; then
        echo "  Cancelled."
        return
    fi

    # Release floating IP first
    local fip_id=$(ibmcloud is floating-ips --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
for ip in data:
    if ip.get('name','') == '${INSTANCE_NAME}-ip':
        print(ip['id'])
        break
" 2>/dev/null)

    if [ -n "$fip_id" ]; then
        echo "  Releasing floating IP..."
        ibmcloud is floating-ip-release "$fip_id" -f 2>/dev/null || true
    fi

    echo "  Destroying instance..."
    ibmcloud is instance-delete "$id" -f
    echo -e "  ${GREEN}Instance destroyed. Billing will stop shortly.${NC}"
}

cmd_create() {
    # Check if already exists
    local existing=$(get_instance_id)
    if [ -n "$existing" ]; then
        echo -e "  ${YELLOW}Instance '$INSTANCE_NAME' already exists.${NC}"
        cmd_status
        return
    fi

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Provisioning Gaudi 3 Instance${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo "  Profile: $PROFILE"
    echo "  Zone: $ZONE"
    echo "  VPC: $VPC_NAME"
    echo "  Subnet: $SUBNET_NAME"
    echo "  Key: $KEY_NAME"
    echo
    echo -e "  ${YELLOW}Estimated cost: ~\$60/hr (apply promo '$PROMO_CODE' for 50% off)${NC}"
    echo
    echo -n "  Proceed? [y/N]: "
    read -r proceed
    if [ "$proceed" != "y" ] && [ "$proceed" != "Y" ]; then
        echo "  Cancelled."
        exit 0
    fi

    # Find Ubuntu 24.04 image
    echo
    echo "  Finding Ubuntu 24.04 image..."
    IMAGE_ID=$(ibmcloud is images --visibility public --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
candidates = [i for i in data if '$IMAGE_PATTERN' in i['name'] and i['status'] == 'available' and 'amd64' in i['name']]
candidates.sort(key=lambda x: x['name'], reverse=True)
if candidates:
    print(candidates[0]['id'])
")

    if [ -z "$IMAGE_ID" ]; then
        fail_exit "Could not find Ubuntu 24.04 image"
    fi
    echo "  Image: $IMAGE_ID"

    # Create instance
    echo "  Creating instance..."
    ibmcloud is instance-create "$INSTANCE_NAME" "$VPC_NAME" "$ZONE" \
        "$PROFILE" "$SUBNET_NAME" \
        --image "$IMAGE_ID" \
        --keys "$KEY_NAME" \
        --output json 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Instance ID: {data['id']}\")
print(f\"  Status: {data['status']}\")
"

    # Wait for running state
    echo "  Waiting for instance to reach 'running' state..."
    for i in $(seq 1 60); do
        status=$(get_instance_status)
        if [ "$status" = "running" ]; then
            echo -e "  ${GREEN}Instance is running!${NC}"
            break
        fi
        echo "    ... $status (${i}/60)"
        sleep 10
    done

    status=$(get_instance_status)
    if [ "$status" != "running" ]; then
        echo -e "  ${RED}Instance did not reach running state (status: $status)${NC}"
        echo "  Check: ibmcloud is instance $INSTANCE_NAME"
        exit 1
    fi

    # Get primary NIC
    echo "  Attaching floating IP..."
    NIC_ID=$(ibmcloud is instance "$INSTANCE_NAME" --output json 2>/dev/null | \
        python3 -c "
import json, sys
data = json.load(sys.stdin)
nics = data.get('network_interfaces', [])
if nics:
    print(nics[0]['id'])
")

    if [ -z "$NIC_ID" ]; then
        echo -e "  ${RED}Could not find network interface${NC}"
        exit 1
    fi

    # Create and bind floating IP
    ibmcloud is floating-ip-reserve "${INSTANCE_NAME}-ip" \
        --nic "$NIC_ID" --in "$INSTANCE_NAME" \
        --output json 2>&1 | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f\"  Floating IP: {data['address']}\")
"

    FLOAT_IP=$(get_floating_ip)
    echo
    echo "  Waiting for SSH to become available..."
    for i in $(seq 1 30); do
        if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
               -o ConnectTimeout=5 root@"$FLOAT_IP" "echo SSH_OK" 2>/dev/null | grep -q SSH_OK; then
            echo -e "  ${GREEN}SSH is ready!${NC}"
            break
        fi
        echo "    ... waiting (${i}/30)"
        sleep 10
    done

    echo
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Instance Ready${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo
    echo "  IP: $FLOAT_IP"
    echo "  SSH: ssh root@$FLOAT_IP"
    echo
    echo "  To start the benchmark:"
    echo "    ssh root@$FLOAT_IP"
    echo "    export ANTHROPIC_API_KEY=sk-ant-..."
    echo "    git clone $REPO_URL $WORK_DIR"
    echo "    cd $WORK_DIR"
    echo "    bash scripts/launch.sh"
    echo
    echo -e "  ${YELLOW}Don't forget: bash scripts/provision.sh --destroy when done!${NC}"
}

# ── Main ─────────────────────────────────────────────────────

case "${1:-create}" in
    --status)   cmd_status ;;
    --cost)     cmd_cost ;;
    --ssh)      cmd_ssh ;;
    --destroy)  cmd_destroy ;;
    *)          cmd_create ;;
esac
