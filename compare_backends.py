#!/usr/bin/env python3
"""
Cross-backend comparison: CUDA vs Apple Silicon results.

Reads results from both autoresearch (Apple Silicon) and autoresearch-cuda repos,
compares val_bpb, tok/sec, MFU across matching datasets.

Usage:
    uv run compare_backends.py [--arm-dir /path/to/autoresearch]
"""

import argparse
import os
import csv
from pathlib import Path

ARM_DEFAULT = os.path.expanduser("~/Claude_Primary/multi-dataset/autoresearch")
CUDA_DIR = Path(__file__).parent


def read_results(results_dir):
    """Read best results per dataset from results/<dataset>/results.tsv files."""
    results = {}
    results_path = Path(results_dir) / "results"
    if not results_path.exists():
        return results

    for dataset_dir in sorted(results_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
        tsv_path = dataset_dir / "results.tsv"
        if not tsv_path.exists():
            continue

        dataset = dataset_dir.name
        best_bpb = float("inf")
        best_row = None

        with open(tsv_path) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                try:
                    bpb = float(row.get("val_bpb", "0"))
                    status = row.get("status", "").strip()
                    if status == "keep" and 0 < bpb < best_bpb:
                        best_bpb = bpb
                        best_row = row
                except (ValueError, TypeError):
                    continue

        if best_row:
            results[dataset] = {
                "val_bpb": best_bpb,
                "memory_gb": float(best_row.get("memory_gb", "0")),
                "description": best_row.get("description", "").strip(),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare CUDA vs Apple Silicon results")
    parser.add_argument("--arm-dir", default=ARM_DEFAULT,
                        help="Path to Apple Silicon autoresearch repo")
    args = parser.parse_args()

    arm_results = read_results(args.arm_dir)
    cuda_results = read_results(CUDA_DIR)

    all_datasets = sorted(set(arm_results.keys()) | set(cuda_results.keys()))

    if not all_datasets:
        print("No results found in either repo.")
        return

    # Header
    print(f"{'Dataset':<20} {'ARM val_bpb':>12} {'CUDA val_bpb':>12} {'Delta':>10} {'Winner':>8}")
    print("-" * 72)

    for ds in all_datasets:
        arm = arm_results.get(ds)
        cuda = cuda_results.get(ds)

        arm_bpb = f"{arm['val_bpb']:.6f}" if arm else "—"
        cuda_bpb = f"{cuda['val_bpb']:.6f}" if cuda else "—"

        if arm and cuda:
            delta = cuda["val_bpb"] - arm["val_bpb"]
            delta_str = f"{delta:+.6f}"
            winner = "CUDA" if delta < 0 else "ARM" if delta > 0 else "TIE"
        else:
            delta_str = "—"
            winner = "CUDA" if cuda else "ARM"

        print(f"{ds:<20} {arm_bpb:>12} {cuda_bpb:>12} {delta_str:>10} {winner:>8}")

    print()

    # Summary
    both = [ds for ds in all_datasets if ds in arm_results and ds in cuda_results]
    if both:
        cuda_wins = sum(1 for ds in both if cuda_results[ds]["val_bpb"] < arm_results[ds]["val_bpb"])
        arm_wins = sum(1 for ds in both if arm_results[ds]["val_bpb"] < cuda_results[ds]["val_bpb"])
        print(f"Datasets compared: {len(both)}")
        print(f"CUDA wins: {cuda_wins}, ARM wins: {arm_wins}, Ties: {len(both) - cuda_wins - arm_wins}")


if __name__ == "__main__":
    main()
