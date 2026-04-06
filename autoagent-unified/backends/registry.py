# Ported from autoresearch-unified/backends/registry.py (MIT)
"""Platform registry — central metadata store for all supported backends."""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent

PLATFORMS = {
    "cuda": {
        "training_script": "platforms/cuda/train_cuda.py",
        "display_name": "NVIDIA CUDA",
        "flops_table": {
            "h100": 756e12, "h200": 756e12, "h800": 756e12,
            "a100": 312e12, "l40s": 362e12, "l40": 181e12,
            "rtx 6000 ada": 363e12, "rtx 4000 ada": 105e12,
            "rtx pro 6000": 380e12, "b200": 2250e12, "b300": 2250e12,
            "rtx 5090": 419e12, "rtx 4090": 330e12, "rtx 4080": 194e12, "rtx 3090": 142e12,
        },
    },
    "rocm": {
        "training_script": "platforms/rocm/train_rocm.py",
        "display_name": "AMD ROCm 6.x",
        "flops_table": {
            "mi350x": 2300e12, "mi325x": 2615e12, "mi308x": 1524e12,
            "mi300x": 1307e12, "mi300a": 980e12, "mi250x": 383e12,
            "mi250": 362e12, "mi210": 181e12, "mi100": 184e12,
            "w7900": 122e12, "rx 7900 xtx": 123e12, "rx 7900 xt": 103e12,
        },
    },
    "rocm7": {
        "training_script": "platforms/rocm/train_rocm7.py",
        "display_name": "AMD ROCm 7.x",
        "flops_table": None,
    },
    "mlx": {
        "training_script": "platforms/metal/train_mlx.py",
        "display_name": "Apple MLX",
        "flops_table": None,
    },
    "mps": {
        "training_script": "platforms/metal/train_mlx.py",
        "display_name": "Apple MPS",
        "flops_table": None,
    },
    "hpu": {
        "training_script": "platforms/gaudi/train_gaudi.py",
        "display_name": "Intel Gaudi HPU",
        "flops_table": {"gaudi 3": 1835e12, "gaudi 2": 865e12},
    },
}


def get_training_script(backend: str) -> str:
    if backend not in PLATFORMS:
        raise ValueError(f"Unknown backend '{backend}'. Available: {', '.join(PLATFORMS.keys())}")
    return PLATFORMS[backend]["training_script"]


def get_display_name(backend: str) -> str:
    if backend in PLATFORMS:
        return PLATFORMS[backend]["display_name"]
    return backend.upper()


def get_platform_flops(backend: str, chip_name: str) -> float | None:
    chip_lower = chip_name.lower()
    effective_backend = "rocm" if backend == "rocm7" else backend
    if effective_backend not in PLATFORMS:
        return None
    table = PLATFORMS[effective_backend].get("flops_table")
    if not table:
        return None
    for key, flops in table.items():
        if key in chip_lower:
            return flops
    return None


def list_backends() -> list[str]:
    return list(PLATFORMS.keys())
