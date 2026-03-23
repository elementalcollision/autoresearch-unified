"""Hardware info for TUI display — supports Gaudi HPU, NVIDIA/AMD GPU, and Apple Silicon."""

import platform
import subprocess
import re


def _get_gaudi_info() -> dict | None:
    """Try to get Intel Gaudi info via hl-smi."""
    try:
        result = subprocess.run(
            ['hl-smi', '-Q', 'name,memory.total', '-f', 'csv,noheader'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            first = lines[0].strip()
            parts = [p.strip() for p in first.split(',')]
            info = {
                'chip_name': parts[0] if parts else 'Gaudi 3',
                'gpu_cores': 64,
                'memory_gb': 128,
                'peak_tflops': 1835.0,
                'device_count': len(lines),
            }
            if len(parts) >= 2:
                m = re.search(r'(\d+)', parts[1])
                if m:
                    info['memory_gb'] = int(m.group(1)) / 1024
            return info
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None


def _get_nvidia_info() -> dict | None:
    """Try to get NVIDIA GPU info via backends module (uses torch.cuda)."""
    try:
        from backends import get_hardware_info, get_peak_flops
        hw = get_hardware_info()
        if hw.get("chip_name", "unknown") != "unknown":
            peak_flops = get_peak_flops(hw)
            return {
                'chip_name': hw['chip_name'],
                'gpu_cores': hw['gpu_cores'],
                'memory_gb': hw['memory_gb'],
                'peak_tflops': peak_flops / 1e12,
            }
    except Exception:
        pass
    return None


def _get_apple_silicon_info() -> dict:
    """Get Apple Silicon hardware info via sysctl."""
    info = {
        'chip_name': 'Unknown',
        'gpu_cores': 0,
        'memory_gb': 0,
        'peak_tflops': 0.0,
    }

    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True, text=True, timeout=5,
        )
        info['chip_name'] = result.stdout.strip()
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'hw.memsize'],
            capture_output=True, text=True, timeout=5,
        )
        info['memory_gb'] = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass

    try:
        result = subprocess.run(
            ['/usr/sbin/sysctl', '-n', 'hw.perflevel0.gpucount'],
            capture_output=True, text=True, timeout=5,
        )
        info['gpu_cores'] = int(result.stdout.strip())
    except Exception:
        chip = info['chip_name'].lower()
        gpu_core_map = {
            ("m1", "base"): 8, ("m1", "pro"): 16, ("m1", "max"): 32, ("m1", "ultra"): 64,
            ("m2", "base"): 10, ("m2", "pro"): 19, ("m2", "max"): 38, ("m2", "ultra"): 76,
            ("m3", "base"): 10, ("m3", "pro"): 18, ("m3", "max"): 40, ("m3", "ultra"): 80,
            ("m4", "base"): 10, ("m4", "pro"): 20, ("m4", "max"): 40, ("m4", "ultra"): 80,
            ("m5", "base"): 10, ("m5", "pro"): 20, ("m5", "max"): 40, ("m5", "ultra"): 80,
        }
        gen_match = re.search(r'(m[1-9])', chip)
        if gen_match:
            gen = gen_match.group(1)
            tier = "ultra" if "ultra" in chip else "max" if "max" in chip else "pro" if "pro" in chip else "base"
            info['gpu_cores'] = gpu_core_map.get((gen, tier), 0)

    chip = info['chip_name'].lower()
    flops_per_core = {
        "m1": 0.5e12, "m2": 0.55e12, "m3": 0.65e12,
        "m4": 0.7e12, "m5": 0.85e12,
    }
    gen_match = re.search(r'(m[1-9])', chip)
    if gen_match and info['gpu_cores'] > 0:
        gen = gen_match.group(1)
        fpc = flops_per_core.get(gen, 0.5e12)
        info['peak_tflops'] = info['gpu_cores'] * fpc / 1e12

    return info


def get_hardware_summary() -> dict:
    """Get hardware info for the current platform.

    Returns dict with: chip_name, gpu_cores, memory_gb, peak_tflops
    """
    # Try Gaudi HPU first
    gaudi = _get_gaudi_info()
    if gaudi:
        return gaudi

    # Try NVIDIA/AMD (works on Linux GPU droplets)
    nvidia = _get_nvidia_info()
    if nvidia:
        return nvidia

    # Fall back to Apple Silicon
    if platform.system() == "Darwin":
        return _get_apple_silicon_info()

    return {
        'chip_name': 'Unknown',
        'gpu_cores': 0,
        'memory_gb': 0,
        'peak_tflops': 0.0,
    }
