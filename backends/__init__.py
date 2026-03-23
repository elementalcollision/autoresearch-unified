"""
Hardware auto-detection, chip tier classification, and hyperparameter suggestions.
Supports all platforms: Intel Gaudi HPU, AMD ROCm, NVIDIA CUDA, Apple MLX, and Apple MPS.
"""

import os
import sys
import subprocess
import re


def get_rocm_version():
    """Return (major, minor) ROCm version tuple, or None if not ROCm."""
    try:
        import torch
        hip_version = getattr(torch.version, 'hip', None)
        if hip_version:
            # hip_version is like "6.3.42134" — extract major.minor
            parts = hip_version.split('.')
            return (int(parts[0]), int(parts[1]))
    except (ImportError, ValueError, IndexError):
        pass
    return None


def detect_backend():
    """
    Auto-detect best available backend.
    Priority: HPU > ROCm > CUDA > MLX > MPS.
    Override with AUTORESEARCH_BACKEND env var: 'hpu', 'rocm', 'rocm7', 'cuda', 'mlx', 'mps', or 'auto'.
    """
    override = os.environ.get("AUTORESEARCH_BACKEND", "auto").lower()
    valid = ("auto", "hpu", "rocm", "rocm7", "cuda", "mlx", "mps")
    if override not in valid:
        raise ValueError(f"AUTORESEARCH_BACKEND must be one of {valid}, got '{override}'")

    # --- Explicit backend requests ---

    if override == "hpu":
        try:
            import torch
            if hasattr(torch, 'hpu') and torch.hpu.is_available():
                return "hpu"
            raise RuntimeError(
                "PyTorch is installed but Gaudi HPU is not available. "
                "Ensure habanalabs drivers are installed and you are in the Habana Docker container."
            )
        except ImportError:
            raise RuntimeError(
                "AUTORESEARCH_BACKEND=hpu but torch is not installed. "
                "Ensure you are running inside the Habana Docker container."
            )

    if override == "rocm7":
        try:
            import torch
            if torch.cuda.is_available():
                return "rocm7"
            raise RuntimeError("PyTorch is installed but ROCm/HIP is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=rocm7 but torch is not installed. "
                               "Install with: pip install torch (ROCm wheels)")

    if override == "rocm":
        try:
            import torch
            if torch.cuda.is_available():
                return "rocm"
            raise RuntimeError("PyTorch is installed but ROCm/HIP is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=rocm but torch is not installed.")

    if override == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            raise RuntimeError("PyTorch is installed but CUDA is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=cuda but torch is not installed.")

    if override == "mlx":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
            raise RuntimeError("MLX is installed but Metal is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mlx but mlx is not installed")

    if override == "mps":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("PyTorch is installed but MPS is not available")
        except ImportError:
            raise RuntimeError("AUTORESEARCH_BACKEND=mps but torch is not installed")

    # --- Auto-detect: HPU > ROCm/CUDA > MLX > MPS ---

    # Intel Gaudi HPU
    try:
        import torch
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return "hpu"
    except ImportError:
        pass

    # NVIDIA CUDA or AMD ROCm (both report via torch.cuda)
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name.lower()
            if any(x in name for x in ["nvidia", "geforce", "rtx", "tesla", "quadro"]):
                return "cuda"
            else:
                # AMD GPU detected via HIP (reports as cuda-compatible)
                return "rocm"
    except ImportError:
        pass

    # Apple Silicon — MLX preferred over MPS
    if sys.platform == "darwin":
        try:
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
        except ImportError:
            pass

        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

    raise RuntimeError(
        "No compatible backend found. Install at least one:\n"
        "  CUDA:  pip install torch  (with NVIDIA GPU)\n"
        "  ROCm:  pip install torch  (with AMD GPU + ROCm)\n"
        "  HPU:   Run inside Habana Docker container\n"
        "  MLX:   pip install mlx  (macOS only)\n"
        "  MPS:   pip install torch  (macOS only)"
    )


def get_hardware_info():
    """
    Returns hardware info dict with keys:
      memory_gb, chip_name, chip_tier, gpu_cores
    Works for Intel Gaudi HPU, AMD GPUs (ROCm/HIP), NVIDIA GPUs, and Apple Silicon.
    """
    info = {
        "memory_gb": 0,
        "chip_name": "unknown",
        "chip_tier": "unknown",
        "gpu_cores": 0,
        "rocm_version": get_rocm_version(),
    }

    # --- Intel Gaudi HPU ---
    try:
        result = subprocess.run(
            ["hl-smi", "-Q", "name,memory.total", "-f", "csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split("\n")
            first = lines[0].strip()
            parts = [p.strip() for p in first.split(",")]
            if len(parts) >= 1:
                info["chip_name"] = parts[0]
            if len(parts) >= 2:
                m = re.search(r"(\d+)", parts[1])
                if m:
                    mem_mib = int(m.group(1))
                    info["memory_gb"] = mem_mib / 1024
            info["chip_tier"] = "gaudi3"
            info["gpu_cores"] = 64  # TPCs per Gaudi 3 device
            info["device_count"] = len(lines)
            return info
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # --- GPU (NVIDIA CUDA or AMD ROCm via HIP) ---
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["chip_name"] = props.name
            info["memory_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_cores"] = props.multi_processor_count

            name_lower = props.name.lower()

            # AMD Instinct / Radeon Pro / Radeon RX
            if any(x in name_lower for x in ["mi350", "mi325", "mi308", "mi300", "mi250", "mi210", "mi100"]):
                info["chip_tier"] = "datacenter"
            elif any(x in name_lower for x in ["w7900", "w7800", "pro w"]):
                info["chip_tier"] = "professional"
            elif any(x in name_lower for x in ["rx 7900", "rx 7800", "rx 7700"]):
                info["chip_tier"] = "consumer"
            # NVIDIA
            elif any(x in name_lower for x in ["h100", "h200", "a100", "h800", "b200", "b300"]):
                info["chip_tier"] = "datacenter"
            elif any(x in name_lower for x in ["l40", "a40", "rtx 6000", "a6000", "rtx pro 6000"]):
                info["chip_tier"] = "professional"
            elif any(x in name_lower for x in ["rtx 4000", "rtx 3000", "rtx 5000"]):
                info["chip_tier"] = "professional"
            elif "rtx" in name_lower:
                info["chip_tier"] = "consumer"
            else:
                info["chip_tier"] = "unknown"
            return info
    except ImportError:
        pass

    # --- Intel Gaudi fallback via torch.hpu ---
    try:
        import torch
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            info["chip_name"] = "Gaudi 3"
            info["memory_gb"] = 128
            info["chip_tier"] = "gaudi3"
            info["gpu_cores"] = 64
            info["device_count"] = torch.hpu.device_count()
            return info
    except (ImportError, Exception):
        pass

    # --- Apple Silicon ---
    if sys.platform == "darwin":
        try:
            mem_bytes = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["memory_gb"] = mem_bytes / (1024 ** 3)
        except (subprocess.CalledProcessError, ValueError):
            pass

        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            info["chip_name"] = brand
        except subprocess.CalledProcessError:
            pass

        chip = info["chip_name"].lower()
        if "ultra" in chip:
            info["chip_tier"] = "ultra"
            info["gpu_cores"] = 80
        elif "max" in chip:
            info["chip_tier"] = "max"
            info["gpu_cores"] = 40
        elif "pro" in chip:
            info["chip_tier"] = "pro"
            info["gpu_cores"] = 18
        elif "m1" in chip or "m2" in chip or "m3" in chip or "m4" in chip or "m5" in chip:
            info["chip_tier"] = "base"
            info["gpu_cores"] = 10
        else:
            info["chip_tier"] = "unknown"

        m = re.search(r"(m[1-5])\s*(pro|max|ultra)?", chip)
        if m:
            gen = m.group(1)
            tier = m.group(2) or "base"
            gpu_core_map = {
                ("m1", "base"): 8, ("m1", "pro"): 16, ("m1", "max"): 32, ("m1", "ultra"): 64,
                ("m2", "base"): 10, ("m2", "pro"): 19, ("m2", "max"): 38, ("m2", "ultra"): 76,
                ("m3", "base"): 10, ("m3", "pro"): 18, ("m3", "max"): 40, ("m3", "ultra"): 76,
                ("m4", "base"): 10, ("m4", "pro"): 20, ("m4", "max"): 40, ("m4", "ultra"): 80,
                ("m5", "base"): 10, ("m5", "pro"): 20, ("m5", "max"): 40, ("m5", "ultra"): 80,
            }
            info["gpu_cores"] = gpu_core_map.get((gen, tier), info["gpu_cores"])

    return info


def get_peak_flops(hw_info=None):
    """
    Estimate peak bf16 TFLOPS for MFU calculation.
    Returns FLOPS (not TFLOPS) for direct use in MFU computation.

    Uses the registry's FLOPS tables first, then falls back to estimation.
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    chip = hw_info["chip_name"].lower()

    # Try registry lookup first (covers all platforms)
    from backends.registry import PLATFORMS
    for backend_name, pdata in PLATFORMS.items():
        table = pdata.get("flops_table")
        if table:
            for key, flops in table.items():
                if key in chip:
                    return flops

    # Fall back to Apple Silicon estimate
    gpu_cores = hw_info["gpu_cores"]
    m = re.search(r"(m[1-5])", chip)
    gen = m.group(1) if m else "m4"

    flops_per_core = {
        "m1": 0.5e12,
        "m2": 0.55e12,
        "m3": 0.65e12,
        "m4": 0.7e12,
        "m5": 0.85e12,
    }.get(gen, 0.65e12)

    return gpu_cores * flops_per_core


def suggest_hyperparameters(hw_info=None):
    """
    Suggest hyperparameters based on hardware tier.
    Returns dict with: depth, device_batch_size, total_batch_size, eval_tokens_multiplier
    """
    if hw_info is None:
        hw_info = get_hardware_info()

    mem_gb = hw_info["memory_gb"]
    tier = hw_info["chip_tier"]

    # Intel Gaudi
    if tier == "gaudi3":
        if mem_gb >= 96:
            return {
                "depth": 16,
                "device_batch_size": 64,
                "total_batch_size": 2**18,  # 256K tokens
                "eval_tokens_multiplier": 10,
            }
        else:
            return {
                "depth": 12,
                "device_batch_size": 32,
                "total_batch_size": 2**17,
                "eval_tokens_multiplier": 10,
            }

    # NVIDIA / AMD datacenter
    if tier == "datacenter":
        return {
            "depth": 12,
            "device_batch_size": 64,
            "total_batch_size": 2**17,  # 128K tokens
            "eval_tokens_multiplier": 10,
        }
    elif tier == "professional":
        return {
            "depth": 10,
            "device_batch_size": 32,
            "total_batch_size": 2**16,  # 64K tokens
            "eval_tokens_multiplier": 10,
        }
    elif tier == "consumer":
        return {
            "depth": 8,
            "device_batch_size": 16,
            "total_batch_size": 2**15,  # 32K tokens
            "eval_tokens_multiplier": 10,
        }

    # Apple Silicon tiers
    if tier == "ultra" or mem_gb >= 128:
        return {
            "depth": 10,
            "device_batch_size": 32,
            "total_batch_size": 2**16,
            "eval_tokens_multiplier": 10,
        }
    elif tier == "max" or mem_gb >= 48:
        return {
            "depth": 8,
            "device_batch_size": 16,
            "total_batch_size": 2**15,
            "eval_tokens_multiplier": 10,
        }
    elif tier == "pro" or mem_gb >= 18:
        return {
            "depth": 6,
            "device_batch_size": 8,
            "total_batch_size": 2**13,
            "eval_tokens_multiplier": 5,
        }
    else:
        return {
            "depth": 4,
            "device_batch_size": 4,
            "total_batch_size": 2**12,
            "eval_tokens_multiplier": 3,
        }


def sync_device(device_type):
    """Synchronize device for accurate timing."""
    if device_type == "hpu":
        import torch
        torch.hpu.synchronize()
    elif device_type == "cuda":
        import torch
        torch.cuda.synchronize()
    elif device_type == "mps":
        import torch
        torch.mps.synchronize()


def get_peak_memory_mb(device_type):
    """Get peak memory usage in MB."""
    if device_type == "hpu":
        import torch
        try:
            return torch.hpu.max_memory_allocated() / 1024 / 1024
        except (AttributeError, RuntimeError):
            return 0.0
    elif device_type == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device_type == "mps":
        import torch
        try:
            return torch.mps.driver_allocated_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    elif device_type == "mlx":
        import mlx.core as mx
        try:
            return mx.get_peak_memory() / 1024 / 1024
        except AttributeError:
            return 0.0
    return 0.0


def print_hardware_summary():
    """Print a summary of detected hardware and suggested config."""
    hw = get_hardware_info()
    hp = suggest_hyperparameters(hw)
    peak_flops = get_peak_flops(hw)

    print(f"Hardware: {hw['chip_name']}")
    print(f"  Memory: {hw['memory_gb']:.0f} GB")
    print(f"  GPU cores/SMs/TPCs: {hw['gpu_cores']}")
    print(f"  Tier: {hw['chip_tier']}")
    print(f"  Peak bf16 FLOPS: {peak_flops:.2e}")
    print(f"Suggested config:")
    print(f"  Depth: {hp['depth']}")
    print(f"  Device batch size: {hp['device_batch_size']}")
    print(f"  Total batch size: {hp['total_batch_size']:,}")


if __name__ == "__main__":
    print_hardware_summary()
    print(f"\nDetected backend: {detect_backend()}")
