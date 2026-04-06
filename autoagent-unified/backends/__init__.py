# Ported from autoresearch-unified/backends/__init__.py (MIT)
"""Hardware auto-detection, chip tier classification, and hyperparameter suggestions.
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
            parts = hip_version.split('.')
            return (int(parts[0]), int(parts[1]))
    except (ImportError, ValueError, IndexError):
        pass
    return None


def detect_backend():
    """Auto-detect best available backend. Priority: HPU > ROCm > CUDA > MLX > MPS."""
    override = os.environ.get("AUTORESEARCH_BACKEND", "auto").lower()
    valid = ("auto", "hpu", "rocm", "rocm7", "cuda", "mlx", "mps")
    if override not in valid:
        raise ValueError(f"AUTORESEARCH_BACKEND must be one of {valid}, got '{override}'")

    if override != "auto":
        # Explicit backend requests
        if override == "hpu":
            import torch
            if hasattr(torch, 'hpu') and torch.hpu.is_available():
                return "hpu"
            raise RuntimeError("Gaudi HPU not available")
        if override in ("rocm", "rocm7", "cuda"):
            import torch
            if torch.cuda.is_available():
                return override
            raise RuntimeError(f"{override} not available")
        if override == "mlx":
            import mlx.core as mx
            if mx.metal.is_available():
                return "mlx"
            raise RuntimeError("MLX Metal not available")
        if override == "mps":
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            raise RuntimeError("MPS not available")

    # Auto-detect
    try:
        import torch
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return "hpu"
    except ImportError:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_properties(0).name.lower()
            if any(x in name for x in ["nvidia", "geforce", "rtx", "tesla", "quadro"]):
                return "cuda"
            else:
                return "rocm"
    except ImportError:
        pass

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

    raise RuntimeError("No compatible backend found.")


def get_hardware_info():
    """Returns hardware info dict: memory_gb, chip_name, chip_tier, gpu_cores."""
    info = {"memory_gb": 0, "chip_name": "unknown", "chip_tier": "unknown", "gpu_cores": 0, "rocm_version": get_rocm_version()}

    # Intel Gaudi HPU
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
                    info["memory_gb"] = int(m.group(1)) / 1024
            info["chip_tier"] = "gaudi3"
            info["gpu_cores"] = 64
            info["device_count"] = len(lines)
            return info
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # GPU (NVIDIA CUDA or AMD ROCm)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["chip_name"] = props.name
            info["memory_gb"] = props.total_memory / (1024 ** 3)
            info["gpu_cores"] = props.multi_processor_count

            name_lower = props.name.lower()
            if any(x in name_lower for x in ["mi350", "mi325", "mi308", "mi300", "mi250", "mi210", "mi100"]):
                info["chip_tier"] = "datacenter"
            elif any(x in name_lower for x in ["h100", "h200", "a100", "h800", "b200", "b300"]):
                info["chip_tier"] = "datacenter"
            elif any(x in name_lower for x in ["l40", "a40", "rtx 6000", "a6000", "rtx pro 6000"]):
                info["chip_tier"] = "professional"
            elif "rtx" in name_lower:
                info["chip_tier"] = "consumer"
            else:
                info["chip_tier"] = "unknown"
            return info
    except ImportError:
        pass

    # Apple Silicon
    if sys.platform == "darwin":
        try:
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            info["memory_gb"] = mem_bytes / (1024 ** 3)
        except (subprocess.CalledProcessError, ValueError):
            pass
        try:
            brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
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
        else:
            info["chip_tier"] = "base"
            info["gpu_cores"] = 10

    return info


def get_peak_flops(hw_info=None):
    """Estimate peak bf16 FLOPS for MFU calculation."""
    if hw_info is None:
        hw_info = get_hardware_info()

    chip = hw_info["chip_name"].lower()

    from backends.registry import PLATFORMS
    for backend_name, pdata in PLATFORMS.items():
        table = pdata.get("flops_table")
        if table:
            for key, flops in table.items():
                if key in chip:
                    return flops

    gpu_cores = hw_info["gpu_cores"]
    m = re.search(r"(m[1-5])", chip)
    gen = m.group(1) if m else "m4"
    flops_per_core = {"m1": 0.5e12, "m2": 0.55e12, "m3": 0.65e12, "m4": 0.7e12, "m5": 0.85e12}.get(gen, 0.65e12)
    return gpu_cores * flops_per_core
