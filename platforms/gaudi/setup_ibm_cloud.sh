#!/bin/bash
# =============================================================================
# IBM Cloud Gaudi 3 Instance Setup
# One-time setup for gx3d-160x1792x8gaudi3 instances
#
# Usage: bash setup_ibm_cloud.sh
# =============================================================================

set -e

echo "============================================"
echo "  Autoresearch — IBM Cloud Gaudi 3 Setup"
echo "============================================"
echo

# --- Step 1: Verify OS ---
echo "[1/7] Verifying operating system..."
. /etc/os-release
echo "  OS: $PRETTY_NAME"
echo "  Kernel: $(uname -r)"
echo

# --- Step 2: Check IOMMU configuration ---
echo "[2/7] Checking IOMMU configuration..."
if grep -q "iommu=pt" /proc/cmdline && grep -q "intel_iommu=on" /proc/cmdline; then
    echo "  IOMMU: correctly configured"
else
    echo "  WARNING: IOMMU not configured. Adding to GRUB..."
    if grep -q "GRUB_CMDLINE_LINUX_DEFAULT" /etc/default/grub; then
        sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 iommu=pt intel_iommu=on"/' /etc/default/grub
    else
        echo 'GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt intel_iommu=on"' | sudo tee -a /etc/default/grub
    fi
    sudo update-grub
    echo "  IOMMU configuration added. A REBOOT IS REQUIRED."
    echo "  Run: sudo reboot"
    echo "  Then re-run this script."
    exit 1
fi
echo

# --- Step 3: Verify Gaudi drivers ---
echo "[3/7] Verifying Gaudi 3 drivers..."
if command -v hl-smi &> /dev/null; then
    echo "  hl-smi found:"
    hl-smi -Q name,memory.total,driver_version -f csv,noheader 2>/dev/null | head -1 | sed 's/^/    /'
    DEVICE_COUNT=$(hl-smi -Q name -f csv,noheader 2>/dev/null | wc -l)
    echo "  Devices detected: $DEVICE_COUNT"
    if [ "$DEVICE_COUNT" -lt 8 ]; then
        echo "  WARNING: Expected 8 Gaudi 3 devices, found $DEVICE_COUNT"
    fi
else
    echo "  ERROR: hl-smi not found. Install habanalabs drivers:"
    echo "    wget https://vault.habana.ai/artifactory/gaudi-installer/1.23.0/habanalabs-installer.sh"
    echo "    chmod +x habanalabs-installer.sh"
    echo "    ./habanalabs-installer.sh install --type base"
    exit 1
fi
echo

# --- Step 4: Source habanalabs environment ---
echo "[4/7] Sourcing habanalabs environment..."
if ls /etc/profile.d/habanalabs*.sh 1> /dev/null 2>&1; then
    source /etc/profile.d/habanalabs*.sh
    echo "  GC_KERNEL_PATH: ${GC_KERNEL_PATH:-not set}"
    echo "  HABANA_LOGS: ${HABANA_LOGS:-not set}"
    echo "  RDMA_CORE_ROOT: ${RDMA_CORE_ROOT:-not set}"
else
    echo "  WARNING: habanalabs profile not found. Environment may not be configured."
fi
echo

# --- Step 5: Verify Docker ---
echo "[5/7] Verifying Docker and Habana container runtime..."
if command -v docker &> /dev/null; then
    echo "  Docker: $(docker --version)"
    # Check for habana runtime
    if docker info 2>/dev/null | grep -q "habana"; then
        echo "  Habana container runtime: configured"
    else
        echo "  WARNING: Habana container runtime not detected."
        echo "  Install: sudo apt install -y habanalabs-container-runtime"
        echo "  Then add to /etc/docker/daemon.json and restart Docker."
    fi
else
    echo "  Docker not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose-v2
    sudo usermod -aG docker $USER
    echo "  Docker installed. You may need to log out and back in for group changes."
fi
echo

# --- Step 6: Pull Habana Docker image ---
echo "[6/7] Pulling Habana PyTorch Docker image..."
IMAGE="vault.habana.ai/gaudi-docker/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest"
if docker image inspect "$IMAGE" &> /dev/null; then
    echo "  Image already pulled: $IMAGE"
else
    echo "  Pulling $IMAGE..."
    docker pull "$IMAGE"
    echo "  Image pulled successfully."
fi
echo

# --- Step 7: Quick HPU test ---
echo "[7/7] Running quick HPU verification..."
docker run --rm --runtime=habana \
    -e HABANA_VISIBLE_DEVICES=all \
    "$IMAGE" \
    python -c "
import torch
import habana_frameworks.torch as htorch
print(f'  PyTorch version: {torch.__version__}')
print(f'  HPU available: {torch.hpu.is_available()}')
print(f'  HPU device count: {torch.hpu.device_count()}')
# Quick bf16 matmul test
a = torch.randn(32, 32, dtype=torch.bfloat16, device='hpu')
b = torch.randn(32, 32, dtype=torch.bfloat16, device='hpu')
c = a @ b
torch.hpu.synchronize()
print(f'  bf16 matmul test: passed')
print('  All checks passed!')
" 2>/dev/null || echo "  WARNING: Quick test failed. Check driver installation."
echo

echo "============================================"
echo "  Setup Complete!"
echo ""
echo "  Next steps:"
echo "    1. Clone the repo"
echo "    2. cd autoresearch-gaudi"
echo "    3. docker compose run prepare    # download data"
echo "    4. docker compose run train      # single training run"
echo "    5. docker compose run agent      # autonomous agent loop"
echo "============================================"
