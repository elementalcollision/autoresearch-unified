#!/usr/bin/env python3
"""Verify Intel Gaudi 3 HPU is accessible and functional.

Checks:
  1. habana_frameworks.torch can be imported
  2. torch.hpu is available
  3. Device count and properties
  4. bf16 matrix multiply works
  5. torch.compile with hpu_backend works

Usage:
    python scripts/verify_hpu.py
    # Or via Docker:
    docker compose run verify
"""

import sys
import time


def check_import():
    """Step 1: Import Habana frameworks."""
    print("1. Importing habana_frameworks.torch...", end=" ", flush=True)
    try:
        import habana_frameworks.torch as htorch
        print("OK")
        return True
    except ImportError as e:
        print(f"FAILED: {e}")
        print("   Make sure you're running inside the Habana Docker container.")
        return False


def check_hpu_available():
    """Step 2: Check HPU availability."""
    print("2. Checking torch.hpu availability...", end=" ", flush=True)
    import torch
    if torch.hpu.is_available():
        count = torch.hpu.device_count()
        print(f"OK ({count} device{'s' if count > 1 else ''})")
        return True
    else:
        print("FAILED: torch.hpu.is_available() returned False")
        return False


def check_device_info():
    """Step 3: Device info."""
    print("3. Device information:")
    import torch
    count = torch.hpu.device_count()
    for i in range(min(count, 2)):  # Show first 2
        name = torch.hpu.get_device_name(i)
        print(f"   Device {i}: {name}")
    if count > 2:
        print(f"   ... and {count - 2} more devices")
    return True


def check_bf16_matmul():
    """Step 4: bf16 matrix multiply."""
    print("4. Testing bf16 matmul on HPU...", end=" ", flush=True)
    import torch
    device = torch.device("hpu")
    try:
        a = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        b = torch.randn(256, 256, dtype=torch.bfloat16, device=device)
        c = a @ b
        torch.hpu.synchronize()
        # Check result is finite
        if torch.isfinite(c).all():
            print(f"OK (result shape: {c.shape}, dtype: {c.dtype})")
            return True
        else:
            print("FAILED: result contains inf/nan")
            return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def check_torch_compile():
    """Step 5: torch.compile with hpu_backend."""
    print("5. Testing torch.compile(backend='hpu_backend')...", end=" ", flush=True)
    import torch

    @torch.compile(backend="hpu_backend")
    def add_mul(x, y):
        return (x + y) * 2.0

    device = torch.device("hpu")
    try:
        x = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        y = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
        result = add_mul(x, y)
        torch.hpu.synchronize()
        expected = (x + y) * 2.0
        if torch.allclose(result, expected, atol=1e-2):
            print("OK")
            return True
        else:
            max_diff = (result - expected).abs().max().item()
            print(f"FAILED: max difference = {max_diff}")
            return False
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def check_autocast():
    """Step 6: Autocast context manager."""
    print("6. Testing torch.amp.autocast(device_type='hpu')...", end=" ", flush=True)
    import torch
    device = torch.device("hpu")
    try:
        x = torch.randn(64, 64, device=device)  # float32
        with torch.amp.autocast(device_type="hpu", dtype=torch.bfloat16):
            y = x @ x.T
        torch.hpu.synchronize()
        if y.dtype == torch.bfloat16:
            print(f"OK (output dtype: {y.dtype})")
            return True
        else:
            print(f"WARNING: expected bf16, got {y.dtype}")
            return True  # Not a hard failure
    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("  Intel Gaudi 3 HPU Verification")
    print("=" * 60)
    print()

    checks = [
        check_import,
        check_hpu_available,
        check_device_info,
        check_bf16_matmul,
        check_torch_compile,
        check_autocast,
    ]

    passed = 0
    failed = 0
    for check in checks:
        try:
            if check():
                passed += 1
            else:
                failed += 1
                if check in (check_import, check_hpu_available):
                    print("\n  FATAL: Cannot continue without HPU access.")
                    break
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            failed += 1

    print()
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"  All {total} checks passed. HPU is ready.")
    else:
        print(f"  {passed}/{total} checks passed, {failed} failed.")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
