"""Hardware profiling tool for autoagent's agent harness.

Reports detected hardware, peak FLOPS, and suggested hyperparameters.
"""

import json

from autoresearch.hardware import get_hardware_summary


def get_hardware_profile() -> str:
    """Return detected hardware, peak FLOPS, suggested hyperparameters.

    Returns:
        JSON string with hardware info and recommendations.
    """
    hw = get_hardware_summary()

    profile = {
        "chip_name": hw.get("chip_name", "Unknown"),
        "gpu_cores": hw.get("gpu_cores", 0),
        "memory_gb": hw.get("memory_gb", 0),
        "peak_tflops": hw.get("peak_tflops", 0.0),
    }

    # Add device count if multi-device
    if "device_count" in hw:
        profile["device_count"] = hw["device_count"]

    return json.dumps(profile, indent=2)
