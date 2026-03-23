#!/usr/bin/env python3
"""Universal training script dispatcher.

Detects the available backend (or reads AUTORESEARCH_BACKEND env var)
and exec's the appropriate platform-specific training script.

Usage:
    python train.py                              # auto-detect backend
    AUTORESEARCH_BACKEND=cuda python train.py    # force CUDA
    AUTORESEARCH_BACKEND=rocm python train.py    # force ROCm 6.x
    AUTORESEARCH_BACKEND=rocm7 python train.py   # force ROCm 7.x
    AUTORESEARCH_BACKEND=hpu python train.py     # force Gaudi HPU
    AUTORESEARCH_BACKEND=mlx python train.py     # force Apple MLX
"""

import os
import sys

# Ensure project root is on sys.path so platform scripts can import backends/
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backends import detect_backend
from backends.registry import get_training_script, get_display_name

backend = detect_backend()
script_path = os.path.join(project_root, get_training_script(backend))

if not os.path.exists(script_path):
    print(f"ERROR: Training script not found: {script_path}", file=sys.stderr)
    sys.exit(1)

print(f"Backend: {get_display_name(backend)} -> {script_path}", flush=True)

# Ensure project root is importable by the platform-specific script
os.environ["PYTHONPATH"] = project_root + os.pathsep + os.environ.get("PYTHONPATH", "")

# Replace this process with the platform-specific training script
os.execv(sys.executable, [sys.executable, script_path] + sys.argv[1:])
