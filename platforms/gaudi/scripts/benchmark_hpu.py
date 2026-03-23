#!/usr/bin/env python3
"""Benchmark Intel Gaudi 3 HPU performance.

Tests:
  1. bf16 matrix multiply throughput at various sizes
  2. Attention (SDPA) throughput
  3. Newton-Schulz orthogonalization (from Muon optimizer)

Usage:
    python scripts/benchmark_hpu.py
"""

import sys
import time

import habana_frameworks.torch as htorch
import torch


def benchmark_matmul():
    """Benchmark bf16 matmul at various sizes."""
    print("\n  bf16 Matrix Multiply Throughput")
    print("  " + "-" * 50)
    device = torch.device("hpu")

    sizes = [512, 1024, 2048, 4096, 8192]
    warmup_iters = 5
    bench_iters = 20

    for n in sizes:
        a = torch.randn(n, n, dtype=torch.bfloat16, device=device)
        b = torch.randn(n, n, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(warmup_iters):
            _ = a @ b
        torch.hpu.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = a @ b
        torch.hpu.synchronize()
        elapsed = time.perf_counter() - start

        flops_per_iter = 2 * n * n * n  # 2*N^3 for matmul
        total_flops = flops_per_iter * bench_iters
        tflops = total_flops / elapsed / 1e12
        ms_per_iter = elapsed / bench_iters * 1000

        print(f"  {n:>5}x{n:<5}  {ms_per_iter:>7.2f} ms/iter  {tflops:>7.1f} TFLOPS")


def benchmark_attention():
    """Benchmark scaled dot-product attention."""
    print("\n  Scaled Dot-Product Attention")
    print("  " + "-" * 50)
    device = torch.device("hpu")

    configs = [
        (4, 128, 256, 64),    # batch, heads, seq_len, head_dim
        (4, 128, 512, 64),
        (4, 128, 1024, 64),
        (8, 128, 512, 64),
    ]
    warmup_iters = 3
    bench_iters = 10

    for batch, heads, seq_len, head_dim in configs:
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(warmup_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.hpu.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.hpu.synchronize()
        elapsed = time.perf_counter() - start

        ms_per_iter = elapsed / bench_iters * 1000
        print(f"  B={batch} H={heads} S={seq_len} D={head_dim}  {ms_per_iter:>7.2f} ms/iter")


def benchmark_newton_schulz():
    """Benchmark Newton-Schulz orthogonalization (used in Muon optimizer)."""
    print("\n  Newton-Schulz Orthogonalization (5 iterations)")
    print("  " + "-" * 50)
    device = torch.device("hpu")

    # Coefficients from the Muon optimizer
    a, b, c = (3.4445, -4.7750, 2.0315)

    sizes = [256, 512, 1024, 2048]
    warmup_iters = 3
    bench_iters = 10

    for n in sizes:
        X = torch.randn(n, n, dtype=torch.bfloat16, device=device)
        X = X / (X.norm() + 1e-7)

        def newton_schulz_5(G):
            assert G.ndim >= 2
            a, b, c = (3.4445, -4.7750, 2.0315)
            X = G.bfloat16()
            if G.size(-2) > G.size(-1):
                X = X.T
            for _ in range(5):
                A = X @ X.T
                B = b * A + c * A @ A
                X = a * X + B @ X
            if G.size(-2) > G.size(-1):
                X = X.T
            return X

        # Warmup
        for _ in range(warmup_iters):
            _ = newton_schulz_5(X)
        torch.hpu.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            _ = newton_schulz_5(X)
        torch.hpu.synchronize()
        elapsed = time.perf_counter() - start

        ms_per_iter = elapsed / bench_iters * 1000
        print(f"  {n:>5}x{n:<5}  {ms_per_iter:>7.2f} ms/iter")


def benchmark_memory():
    """Check peak memory allocation."""
    print("\n  Memory Allocation Test")
    print("  " + "-" * 50)
    device = torch.device("hpu")

    # Allocate progressively larger tensors
    sizes_gb = [1, 4, 16, 64]
    for gb in sizes_gb:
        try:
            n_elements = gb * 1024 * 1024 * 1024 // 2  # bf16 = 2 bytes
            t = torch.empty(n_elements, dtype=torch.bfloat16, device=device)
            torch.hpu.synchronize()
            actual_gb = t.nelement() * 2 / (1024**3)
            print(f"  {gb:>3} GB allocation: OK ({actual_gb:.1f} GB)")
            del t
            torch.hpu.synchronize()
        except Exception as e:
            print(f"  {gb:>3} GB allocation: FAILED ({e})")
            break

    # Report peak
    peak_mb = torch.hpu.max_memory_allocated() / (1024 * 1024)
    print(f"\n  Peak memory allocated: {peak_mb:.0f} MB")


def main():
    print("=" * 60)
    print("  Intel Gaudi 3 HPU Benchmark")
    print("=" * 60)

    device_name = torch.hpu.get_device_name(0)
    device_count = torch.hpu.device_count()
    print(f"\n  Device: {device_name}")
    print(f"  Count:  {device_count}")

    try:
        benchmark_matmul()
        benchmark_attention()
        benchmark_newton_schulz()
        benchmark_memory()
    except Exception as e:
        print(f"\n  BENCHMARK ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  Benchmark complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
