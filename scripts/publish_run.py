#!/usr/bin/env python3
"""
Publish a completed autoresearch run to the unified repo.

Handles the full pipeline:
1. Copy raw results from WSL
2. Validate experiment count
3. Convert to 14-column unified format
4. Truncate to exactly 100 experiments
5. Rebuild benchmark
6. Commit and push to the correct branch

Usage:
    python scripts/publish_run.py --model kimik25 --run 3
    python scripts/publish_run.py --model kimik25 --run 3 --dry-run
    python scripts/publish_run.py --model qwen35-397b --run 2 --gpu "NVIDIA GeForce RTX 5090"
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

# Defaults
WSL_DISTRO = "Ubuntu-22.04"
WSL_RESULTS = "/home/pat/autoresearch/agent_results_pubmed.tsv"
WSL_LOG = "/home/pat/autoresearch/agent_pubmed.log"
REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pubmed" / "bmdhodl-rtx5070ti"
CONVERTER = REPO_ROOT / "scripts" / "convert_agent_tsv.py"
BENCHMARK = REPO_ROOT / "scripts" / "build_benchmark.py"

# Model configs
MODELS = {
    "kimik25": {
        "branch": "data/controlled-kimik25-rtx5070ti",
        "gpu": "NVIDIA GeForce RTX 5070 Ti",
        "tag_prefix": "controlled-kimi-k25",
    },
    "qwen35-397b": {
        "branch": "data/controlled-qwen35-397b-rtx5090",
        "gpu": "NVIDIA GeForce RTX 5090",
        "tag_prefix": "controlled-qwen35",
    },
    "sonnet46": {
        "branch": "data/controlled-sonnet46-rtx5070ti",
        "gpu": "NVIDIA GeForce RTX 5070 Ti",
        "tag_prefix": "controlled-sonnet46",
    },
    "gpt41": {
        "branch": "data/controlled-gpt41-rtx5070ti",
        "gpu": "NVIDIA GeForce RTX 5070 Ti",
        "tag_prefix": "controlled-gpt41",
    },
}

TARGET_EXPERIMENTS = 100


def run(cmd, check=True, capture=False):
    """Run a shell command."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    if check and result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if capture and result.stderr:
            print(f"  stderr: {result.stderr.strip()}")
        sys.exit(1)
    return result


def wsl(cmd):
    """Run a command in WSL."""
    return run(f'wsl -d {WSL_DISTRO} bash -c "{cmd}"', capture=True)


def main():
    parser = argparse.ArgumentParser(description="Publish autoresearch run to unified repo")
    parser.add_argument("--model", required=True, choices=MODELS.keys(), help="Model slug")
    parser.add_argument("--run", required=True, type=int, help="Run number (1, 2, 3)")
    parser.add_argument("--gpu", help="Override GPU name")
    parser.add_argument("--wsl-path", default=WSL_RESULTS, help="Path to raw TSV in WSL")
    parser.add_argument("--dry-run", action="store_true", help="Validate only, don't commit/push")
    parser.add_argument("--skip-copy", action="store_true", help="Skip WSL copy (use existing raw file)")
    args = parser.parse_args()

    config = MODELS[args.model]
    gpu = args.gpu or config["gpu"]
    tag = f"{config['tag_prefix']}-r{args.run}"
    branch = config["branch"]
    raw_file = REPO_ROOT / f"results_{args.model}_r{args.run}_raw.tsv"
    output_file = RESULTS_DIR / f"results_{args.model}_r{args.run}.tsv"

    print("=" * 60)
    print(f"  Publishing: {args.model} R{args.run}")
    print(f"  Branch: {branch}")
    print(f"  Tag: {tag}")
    print(f"  GPU: {gpu}")
    print("=" * 60)
    print()

    # Step 1: Checkout branch
    print("[1/8] Checkout branch...")
    os.chdir(REPO_ROOT)
    run(f"git checkout {branch}")
    run("git pull origin " + branch, check=False)
    print()

    # Step 2: Copy from WSL
    if not args.skip_copy:
        print("[2/8] Copy results from WSL...")
        result = wsl(f"wc -l {args.wsl_path}")
        lines = result.stdout.strip().split()[0]
        print(f"  Raw file: {lines} lines in WSL")
        wsl(f"cp {args.wsl_path} /mnt/{str(raw_file).replace(os.sep, '/').replace('C:', 'c')}")
        print(f"  Copied to: {raw_file}")
    else:
        print("[2/8] Skipping WSL copy (--skip-copy)")
        if not raw_file.exists():
            print(f"  ERROR: {raw_file} not found")
            sys.exit(1)
    print()

    # Step 3: Validate raw file
    print("[3/8] Validate raw file...")
    with open(raw_file, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()
    header = raw_lines[0]
    data_rows = raw_lines[1:]
    print(f"  Header columns: {len(header.strip().split(chr(9)))}")
    print(f"  Data rows: {len(data_rows)}")

    if len(data_rows) < TARGET_EXPERIMENTS:
        print(f"  WARNING: Only {len(data_rows)} experiments (expected {TARGET_EXPERIMENTS})")
        print(f"  Run may still be in progress. Continue? (y/N)")
        if input().strip().lower() != "y":
            sys.exit(0)
    elif len(data_rows) > TARGET_EXPERIMENTS:
        print(f"  TRIMMING: {len(data_rows)} -> {TARGET_EXPERIMENTS} rows")
        raw_lines = [header] + data_rows[:TARGET_EXPERIMENTS]
        with open(raw_file, "w", encoding="utf-8") as f:
            f.writelines(raw_lines)
        data_rows = data_rows[:TARGET_EXPERIMENTS]
    print()

    # Step 4: Count statuses
    print("[4/8] Analyze results...")
    statuses = {}
    for row in data_rows:
        cols = row.strip().split("\t")
        status = cols[3] if len(cols) > 3 else "unknown"
        statuses[status] = statuses.get(status, 0) + 1
    for s, c in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    total = sum(statuses.values())
    keeps = statuses.get("keep", 0) - (1 if "baseline" not in statuses else 0)
    crashes = statuses.get("crash", 0)
    keep_rate = keeps / (total - 1) * 100 if total > 1 else 0  # exclude baseline
    crash_rate = crashes / (total - 1) * 100 if total > 1 else 0
    print(f"  ---")
    print(f"  Total: {total} | Keep rate: {keep_rate:.1f}% | Crash rate: {crash_rate:.1f}%")

    # Find best val_bpb
    best_bpb = float("inf")
    for row in data_rows:
        cols = row.strip().split("\t")
        if len(cols) > 1:
            try:
                bpb = float(cols[1])
                if bpb > 0 and bpb < best_bpb:
                    best_bpb = bpb
            except ValueError:
                pass
    print(f"  Best val_bpb: {best_bpb:.6f}")
    print()

    # Step 5: Convert
    print("[5/8] Convert to 14-column format...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run(f'python "{CONVERTER}" "{raw_file}" "{output_file}" --tag {tag} --gpu "{gpu}"')

    # Verify output has exactly 100 data rows
    with open(output_file, "r") as f:
        out_lines = len(f.readlines())
    print(f"  Output: {out_lines} lines (header + {out_lines - 1} data)")
    if out_lines - 1 != TARGET_EXPERIMENTS:
        print(f"  ERROR: Expected {TARGET_EXPERIMENTS} data rows, got {out_lines - 1}")
        # Force truncate
        with open(output_file, "r") as f:
            all_lines = f.readlines()
        with open(output_file, "w") as f:
            f.writelines(all_lines[:TARGET_EXPERIMENTS + 1])
        print(f"  FIXED: Truncated to {TARGET_EXPERIMENTS} rows")
    print()

    # Step 6: Rebuild benchmark
    print("[6/8] Rebuild benchmark...")
    run(f'python "{BENCHMARK}"')
    print()

    # Step 7: Clean up raw file
    raw_file.unlink(missing_ok=True)

    if args.dry_run:
        print("[7/8] DRY RUN - skipping commit")
        print("[8/8] DRY RUN - skipping push")
        print()
        print("=" * 60)
        print("  DRY RUN COMPLETE - no changes committed")
        print("=" * 60)
        return

    # Step 7: Commit
    print("[7/8] Commit...")
    run(f'git add "{output_file}" docs/benchmark/data.json')
    commit_msg = (
        f"Data: {args.model} R{args.run} ({TARGET_EXPERIMENTS} experiments, {gpu.split()[-1]})\n\n"
        f"Keep rate: {keep_rate:.1f}% ({keeps}/{total-1})\n"
        f"Crash rate: {crash_rate:.1f}% ({crashes}/{total-1})\n"
        f"Best val_bpb: {best_bpb:.6f}\n\n"
        f"Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
    )
    run(f'git commit -m "{commit_msg}"')
    print()

    # Step 8: Push
    print("[8/8] Push...")
    run(f"git push origin {branch}")
    print()

    print("=" * 60)
    print(f"  PUBLISHED: {args.model} R{args.run}")
    print(f"  Keep rate: {keep_rate:.1f}% | Best: {best_bpb:.6f}")
    print(f"  Branch: {branch}")
    print("=" * 60)


if __name__ == "__main__":
    main()
