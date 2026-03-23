#!/usr/bin/env python3
"""
Consolidate results.tsv files from all autoresearch platform repos
into unified Parquet files for HuggingFace dataset upload.

Walks the results directories of:
  - autoresearch-cuda (NVIDIA)
  - ROCm (AMD)
  - multi-dataset/autoresearch (Apple Metal)

Produces:
  - data/experiments.parquet  (all experiment rows with platform metadata)
  - data/hardware.parquet     (GPU hardware reference table)
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Base directory for all repos
DEFAULT_BASE = Path(__file__).resolve().parent.parent.parent  # Claude_Primary/

# GPU metadata derived from directory path patterns
GPU_MAPPINGS = {
    # CUDA repo: results/{dataset}/{gpu_dir}/results.tsv
    "do-rtx4000": {
        "platform": "nvidia_cuda",
        "gpu_name": "RTX 4000 Ada",
        "gpu_provider": "digitalocean",
    },
    "vultr-a100": {
        "platform": "nvidia_cuda",
        "gpu_name": "A100 40GB",
        "gpu_provider": "vultr",
    },
    "runpod-rtxpro6000": {
        "platform": "nvidia_cuda",
        "gpu_name": "RTX Pro 6000 Blackwell",
        "gpu_provider": "runpod",
    },
    # Numbered runs in CUDA repo are earlier DO RTX 4000 runs
    "run1": {
        "platform": "nvidia_cuda",
        "gpu_name": "RTX 4000 Ada",
        "gpu_provider": "digitalocean",
    },
    "run2": {
        "platform": "nvidia_cuda",
        "gpu_name": "RTX 4000 Ada",
        "gpu_provider": "digitalocean",
    },
}

# Directories to skip (sync copies / known duplicates of GPU-specific dirs)
SKIP_PATTERNS = {
    "do-rtx4000-sync",
    "vultr-a100-sync",
}

# Expected TSV columns
EXPECTED_COLUMNS = [
    "exp", "description", "val_bpb", "peak_mem_gb",
    "tok_sec", "mfu", "steps", "status", "notes",
]


def find_results_files(base: Path) -> list[dict]:
    """Discover all results.tsv files and classify them by platform/GPU."""
    sources = []

    # --- CUDA repo ---
    cuda_results = base / "autoresearch-cuda" / "results"
    if cuda_results.exists():
        for tsv in cuda_results.rglob("results.tsv"):
            rel = tsv.relative_to(cuda_results)
            parts = rel.parts  # e.g. ('climbmix', 'do-rtx4000', 'results.tsv')

            # Skip sync directories
            if any(skip in str(rel) for skip in SKIP_PATTERNS):
                continue

            # Skip top-level dataset results.tsv when GPU-specific subdirs exist
            if len(parts) == 2:
                # This is results/{dataset}/results.tsv (no GPU subdir)
                dataset = parts[0]
                dataset_dir = cuda_results / dataset
                has_gpu_subdirs = any(
                    (dataset_dir / gpu_dir).exists()
                    for gpu_dir in GPU_MAPPINGS
                    if gpu_dir not in ("run1", "run2")
                )
                if has_gpu_subdirs:
                    continue  # Skip — duplicate of GPU-specific file
                # No GPU subdirs; treat as standalone DO RTX 4000 result
                sources.append({
                    "path": tsv,
                    "dataset": dataset,
                    "platform": "nvidia_cuda",
                    "gpu_name": "RTX 4000 Ada",
                    "gpu_provider": "digitalocean",
                    "agent_model": "sonnet-4.0",
                    "run_id": "default",
                })
                continue

            if len(parts) >= 3:
                dataset = parts[0]
                gpu_dir = parts[1]

                if gpu_dir in GPU_MAPPINGS:
                    meta = GPU_MAPPINGS[gpu_dir]
                    sources.append({
                        "path": tsv,
                        "dataset": dataset,
                        **meta,
                        "agent_model": "sonnet-4.0",
                        "run_id": gpu_dir,
                    })

    # --- ROCm repo ---
    rocm_results = base / "ROCm" / "results"
    if rocm_results.exists():
        for tsv in rocm_results.rglob("results.tsv"):
            rel = tsv.relative_to(rocm_results)
            parts = rel.parts
            if len(parts) >= 2:
                dataset = parts[0]
                sources.append({
                    "path": tsv,
                    "dataset": dataset,
                    "platform": "amd_rocm",
                    "gpu_name": "MI300X",
                    "gpu_provider": "runpod",
                    "agent_model": "sonnet-4.0",
                    "run_id": "default",
                })

    # --- Apple Metal repo ---
    metal_results = base / "multi-dataset" / "autoresearch" / "results"
    if metal_results.exists():
        for tsv in metal_results.rglob("results.tsv"):
            rel = tsv.relative_to(metal_results)
            parts = rel.parts

            # Detect sonnet-4-6 subdirectory
            if "sonnet-4-6" in parts:
                idx = parts.index("sonnet-4-6")
                dataset = parts[idx + 1] if idx + 1 < len(parts) - 1 else parts[-2]
                agent_model = "sonnet-4.6"
                run_id = "sonnet-4-6"
            else:
                dataset = parts[0]
                agent_model = "sonnet-4.0"
                run_id = "default"

            sources.append({
                "path": tsv,
                "dataset": dataset,
                "platform": "apple_metal",
                "gpu_name": "M5 Max",
                "gpu_provider": "local",
                "agent_model": agent_model,
                "run_id": run_id,
            })

    return sources


def read_tsv(path: Path) -> pd.DataFrame | None:
    """Read a results.tsv file, handling edge cases."""
    if path.stat().st_size == 0:
        print(f"  SKIP: {path} is empty", file=sys.stderr)
        return None

    df = pd.read_csv(path, sep="\t", dtype=str)

    if df.empty:
        print(f"  SKIP: {path} has no data rows", file=sys.stderr)
        return None

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Validate expected columns exist
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        print(f"  WARNING: {path} missing columns: {missing}", file=sys.stderr)

    return df


def deduplicate(sources: list[dict]) -> list[dict]:
    """Remove exact file duplicates based on content hash."""
    seen_hashes = {}
    deduped = []

    for src in sources:
        content = src["path"].read_bytes()
        h = hashlib.md5(content).hexdigest()

        if h in seen_hashes:
            print(
                f"  DEDUP: Skipping {src['path']} "
                f"(identical to {seen_hashes[h]})",
                file=sys.stderr,
            )
            continue

        seen_hashes[h] = src["path"]
        deduped.append(src)

    return deduped


def consolidate(base: Path, output_dir: Path) -> pd.DataFrame:
    """Main consolidation pipeline."""
    print("Discovering results files...")
    sources = find_results_files(base)
    print(f"  Found {len(sources)} results.tsv files")

    print("Deduplicating...")
    sources = deduplicate(sources)
    print(f"  {len(sources)} unique files after deduplication")

    all_rows = []
    for src in sources:
        print(f"  Reading: {src['path'].relative_to(base)}")
        df = read_tsv(src["path"])
        if df is None:
            continue

        # Add metadata columns
        df["platform"] = src["platform"]
        df["gpu_name"] = src["gpu_name"]
        df["gpu_provider"] = src["gpu_provider"]
        df["dataset"] = src["dataset"]
        df["agent_model"] = src["agent_model"]
        df["run_id"] = src["run_id"]

        # Generate experiment_id
        df["experiment_id"] = (
            df["platform"] + "_"
            + df["gpu_name"].str.replace(" ", "-") + "_"
            + df["dataset"] + "_"
            + df["run_id"] + "_"
            + df["exp"]
        )

        all_rows.append(df)

    if not all_rows:
        print("ERROR: No data found!", file=sys.stderr)
        sys.exit(1)

    unified = pd.concat(all_rows, ignore_index=True)

    # Cast numeric columns
    numeric_casts = {
        "val_bpb": "float64",
        "peak_mem_gb": "float32",
        "tok_sec": "float64",
        "mfu": "float32",
        "steps": "float64",
    }
    for col, dtype in numeric_casts.items():
        if col in unified.columns:
            unified[col] = pd.to_numeric(unified[col], errors="coerce").astype(dtype)

    # Reorder columns
    col_order = [
        "experiment_id", "platform", "gpu_name", "gpu_provider", "dataset",
        "agent_model", "run_id", "exp", "description", "val_bpb",
        "peak_mem_gb", "tok_sec", "mfu", "steps", "status", "notes",
    ]
    # Keep only columns that exist
    col_order = [c for c in col_order if c in unified.columns]
    unified = unified[col_order]

    # Write experiments parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_path = output_dir / "experiments.parquet"
    unified.to_parquet(experiments_path, index=False, engine="pyarrow")
    print(f"\nWrote {len(unified)} rows to {experiments_path}")

    # Summary stats
    print("\n--- Summary ---")
    print(f"Total experiments: {len(unified)}")
    print(f"Platforms: {unified['platform'].nunique()}")
    print(f"GPUs: {unified['gpu_name'].nunique()}")
    print(f"Datasets: {unified['dataset'].nunique()}")
    print(f"\nBy platform:")
    print(unified.groupby("platform").size().to_string())
    print(f"\nBy GPU:")
    print(unified.groupby(["platform", "gpu_name", "run_id"]).size().to_string())
    print(f"\nBy dataset:")
    print(unified.groupby("dataset").size().to_string())

    return unified


def build_hardware_table(output_dir: Path):
    """Build hardware reference table from known specs."""
    hardware = [
        {
            "gpu_name": "M1 Max",
            "platform": "apple_metal",
            "architecture": "Apple GPU",
            "compute_units": "32 GPU cores",
            "vram_gb": 64,
            "bf16_tflops": 10.0,
            "memory_bandwidth_gbps": 400,
            "tdp_watts": 60,
            "provider": "local",
            "cost_per_hour": 0.0,
        },
        {
            "gpu_name": "M4 Pro",
            "platform": "apple_metal",
            "architecture": "Apple GPU",
            "compute_units": "16 GPU cores",
            "vram_gb": 24,
            "bf16_tflops": 7.0,
            "memory_bandwidth_gbps": 273,
            "tdp_watts": 40,
            "provider": "local",
            "cost_per_hour": 0.0,
        },
        {
            "gpu_name": "M5 Max",
            "platform": "apple_metal",
            "architecture": "Apple GPU",
            "compute_units": "40 GPU cores",
            "vram_gb": 64,
            "bf16_tflops": 14.0,
            "memory_bandwidth_gbps": 400,
            "tdp_watts": 92,
            "provider": "local",
            "cost_per_hour": 0.0,
        },
        {
            "gpu_name": "RTX 4000 Ada",
            "platform": "nvidia_cuda",
            "architecture": "Ada Lovelace",
            "compute_units": "48 SMs",
            "vram_gb": 20,
            "bf16_tflops": 105.0,
            "memory_bandwidth_gbps": 280,
            "tdp_watts": 130,
            "provider": "digitalocean",
            "cost_per_hour": 0.76,
        },
        {
            "gpu_name": "A100 40GB",
            "platform": "nvidia_cuda",
            "architecture": "Ampere",
            "compute_units": "108 SMs",
            "vram_gb": 40,
            "bf16_tflops": 156.0,
            "memory_bandwidth_gbps": 1555,
            "tdp_watts": 300,
            "provider": "vultr",
            "cost_per_hour": 1.20,
        },
        {
            "gpu_name": "RTX Pro 6000 Blackwell",
            "platform": "nvidia_cuda",
            "architecture": "Blackwell",
            "compute_units": "96 SMs",
            "vram_gb": 96,
            "bf16_tflops": 260.0,
            "memory_bandwidth_gbps": 1100,
            "tdp_watts": 250,
            "provider": "runpod",
            "cost_per_hour": 1.69,
        },
        {
            "gpu_name": "MI300X",
            "platform": "amd_rocm",
            "architecture": "CDNA 3",
            "compute_units": "304 CUs",
            "vram_gb": 192,
            "bf16_tflops": 1307.0,
            "memory_bandwidth_gbps": 5300,
            "tdp_watts": 750,
            "provider": "runpod",
            "cost_per_hour": 1.99,
        },
        {
            "gpu_name": "Gaudi 3",
            "platform": "intel_gaudi",
            "architecture": "Gaudi 3 HPU",
            "compute_units": "64 MME",
            "vram_gb": 128,
            "bf16_tflops": 1835.0,
            "memory_bandwidth_gbps": 3700,
            "tdp_watts": 900,
            "provider": "cloud",
            "cost_per_hour": 0.0,  # TBD
        },
    ]

    df = pd.DataFrame(hardware)
    output_path = output_dir / "hardware.parquet"
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"\nWrote {len(df)} hardware entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Consolidate autoresearch results")
    parser.add_argument(
        "--base", type=Path, default=DEFAULT_BASE,
        help="Base directory containing all autoresearch repos",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for Parquet files",
    )
    args = parser.parse_args()

    print(f"Base directory: {args.base}")
    print(f"Output directory: {args.output}")

    consolidate(args.base, args.output)
    build_hardware_table(args.output)
    print("\nDone!")


if __name__ == "__main__":
    main()
