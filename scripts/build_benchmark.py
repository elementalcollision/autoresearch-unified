#!/usr/bin/env python3
"""Build benchmark leaderboard JSON from TSV result files.

Reads all data/results/{dataset}/{contributor-gpu}/results_*.tsv files,
computes aggregated statistics per model + hardware + dataset combination,
and writes docs/benchmark/data.json for the static leaderboard page.

Usage:
    python scripts/build_benchmark.py
"""

import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "results"
OUTPUT_DIR = ROOT / "docs" / "benchmark"
OUTPUT_FILE = OUTPUT_DIR / "data.json"

# Map filename model slugs to display names
MODEL_NAMES = {
    "sonnet46": "Sonnet 4.6",
    "sonnet4": "Sonnet 4",
    "sonnet37": "Sonnet 3.7",
    "opus4": "Opus 4",
    "gpt41": "GPT-4.1",
    "gpt4o": "GPT-4o",
    "gpt45": "GPT-4.5",
    "o3": "o3",
    "o4mini": "o4-mini",
    "gemini25pro": "Gemini 2.5 Pro",
    "deepseek": "DeepSeek V3",
}

# Map contributor handles to provider context where known
CONTRIBUTOR_PROVIDERS = {
    "bmdhodl": "",           # direct API
    "elementalcollision": "",  # may use OpenRouter
}


def parse_model_from_filename(filename: str) -> str:
    """Extract model slug from filename like results_sonnet46_r1.tsv."""
    match = re.match(r"results_(.+?)_r\d+\.tsv", filename)
    if match:
        return match.group(1)
    return filename.replace("results_", "").replace(".tsv", "")


def parse_contributor_gpu(dirname: str) -> tuple[str, str]:
    """Parse 'bmdhodl-rtx5070ti' into ('bmdhodl', 'rtx5070ti')."""
    parts = dirname.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return dirname, "unknown"


def get_model_display_name(slug: str) -> str:
    """Convert model slug to display name."""
    return MODEL_NAMES.get(slug, slug)


def get_model_family(slug: str) -> str:
    """Determine model family from slug."""
    if slug.startswith("sonnet") or slug.startswith("opus") or slug.startswith("haiku"):
        return "Claude"
    if slug.startswith("gpt") or slug.startswith("o3") or slug.startswith("o4"):
        return "OpenAI"
    if slug.startswith("gemini"):
        return "Google"
    if slug.startswith("deepseek"):
        return "DeepSeek"
    return "Other"


def parse_tsv(filepath: Path) -> list[dict]:
    """Parse a TSV file into list of row dicts."""
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def compute_run_stats(rows: list[dict]) -> dict:
    """Compute stats for a single run (one TSV file)."""
    total_exps = 0
    keeps = 0
    crashes = 0
    discards = 0
    skips = 0
    best_bpb = None
    baseline_bpb = None
    bpb_trajectory = []
    kept_improvements = []

    for row in rows:
        status = (row.get("status") or "").strip().lower()
        val_bpb_str = (row.get("val_bpb") or "0").strip()
        exp = (row.get("exp") or "").strip()
        description = (row.get("description") or "").strip()

        try:
            val_bpb = float(val_bpb_str)
        except (ValueError, TypeError):
            val_bpb = 0.0

        if status == "baseline":
            baseline_bpb = val_bpb
            bpb_trajectory.append({"exp": exp, "val_bpb": val_bpb, "status": status})
            continue

        total_exps += 1

        if status == "keep":
            keeps += 1
            if val_bpb > 0:
                if best_bpb is None or val_bpb < best_bpb:
                    best_bpb = val_bpb
                kept_improvements.append({
                    "exp": exp,
                    "val_bpb": val_bpb,
                    "description": description[:200],
                })
        elif status == "crash":
            crashes += 1
        elif status == "discard":
            discards += 1
        elif status == "skip":
            skips += 1

        bpb_trajectory.append({"exp": exp, "val_bpb": val_bpb, "status": status})

    keep_rate = (keeps / total_exps * 100) if total_exps > 0 else 0
    crash_rate = (crashes / total_exps * 100) if total_exps > 0 else 0

    return {
        "total_exps": total_exps,
        "keeps": keeps,
        "crashes": crashes,
        "discards": discards,
        "skips": skips,
        "keep_rate": keep_rate,
        "crash_rate": crash_rate,
        "best_bpb": best_bpb,
        "baseline_bpb": baseline_bpb,
        "bpb_trajectory": bpb_trajectory,
        "kept_improvements": kept_improvements,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def build_leaderboard():
    """Main build: scan all TSVs, aggregate, write JSON."""
    if not RESULTS_DIR.exists():
        print(f"ERROR: {RESULTS_DIR} not found", file=sys.stderr)
        sys.exit(1)

    # Structure: groups[dataset][model_slug][contributor-gpu] = [run_stats, ...]
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_runs = []

    for dataset_dir in sorted(RESULTS_DIR.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name

        for contrib_dir in sorted(dataset_dir.iterdir()):
            if not contrib_dir.is_dir():
                continue
            contributor, gpu_slug = parse_contributor_gpu(contrib_dir.name)

            for tsv_file in sorted(contrib_dir.glob("results_*.tsv")):
                model_slug = parse_model_from_filename(tsv_file.name)
                rows = parse_tsv(tsv_file)
                if not rows:
                    continue

                stats = compute_run_stats(rows)
                # Get GPU name from data if available
                gpu_name = gpu_slug
                for row in rows:
                    gn = row.get("gpu_name", "").strip()
                    if gn:
                        gpu_name = gn
                        break

                run_info = {
                    "file": tsv_file.relative_to(ROOT).as_posix(),
                    "dataset": dataset,
                    "model_slug": model_slug,
                    "model": get_model_display_name(model_slug),
                    "model_family": get_model_family(model_slug),
                    "contributor": contributor,
                    "gpu_slug": gpu_slug,
                    "gpu_name": gpu_name,
                    **stats,
                }
                all_runs.append(run_info)

                key = f"{model_slug}|{gpu_name}"
                groups[dataset][key][contributor].append(run_info)

    # Build aggregated leaderboard entries
    leaderboard = []

    for dataset in sorted(groups):
        for model_hw_key in groups[dataset]:
            for contributor, runs in groups[dataset][model_hw_key].items():
                n = len(runs)
                keep_rates = [r["keep_rate"] for r in runs]
                crash_rates = [r["crash_rate"] for r in runs]
                best_bpbs = [r["best_bpb"] for r in runs if r["best_bpb"] is not None]

                entry = {
                    "dataset": dataset,
                    "model": runs[0]["model"],
                    "model_slug": runs[0]["model_slug"],
                    "model_family": runs[0]["model_family"],
                    "gpu_name": runs[0]["gpu_name"],
                    "gpu_slug": runs[0]["gpu_slug"],
                    "contributor": contributor,
                    "runs": n,
                    "provisional": n < 3,
                    "keep_rate_mean": round(mean(keep_rates), 1),
                    "keep_rate_sd": round(sd(keep_rates), 1) if n >= 3 else None,
                    "crash_rate_mean": round(mean(crash_rates), 1),
                    "crash_rate_sd": round(sd(crash_rates), 1) if n >= 3 else None,
                    "best_bpb": round(min(best_bpbs), 6) if best_bpbs else None,
                    "total_exps_sum": sum(r["total_exps"] for r in runs),
                    "total_keeps_sum": sum(r["keeps"] for r in runs),
                    "total_crashes_sum": sum(r["crashes"] for r in runs),
                    "run_details": runs,
                }
                leaderboard.append(entry)

    # Sort by keep rate descending (primary), best_bpb ascending (secondary)
    leaderboard.sort(
        key=lambda e: (-e["keep_rate_mean"], e["best_bpb"] or 999)
    )

    # Assign ranks
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1

    # Collect filter options
    datasets = sorted(set(e["dataset"] for e in leaderboard))
    model_families = sorted(set(e["model_family"] for e in leaderboard))
    hardware = sorted(set(e["gpu_name"] for e in leaderboard))

    output = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_runs": len(all_runs),
        "filters": {
            "datasets": datasets,
            "model_families": model_families,
            "hardware": hardware,
        },
        "leaderboard": leaderboard,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Only write if data actually changed (ignore generated_at timestamp)
    # This avoids spurious commits in CI when only the timestamp differs.
    skip_write = False
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
        existing_data = {k: v for k, v in existing.items() if k != "generated_at"}
        new_data = {k: v for k, v in output.items() if k != "generated_at"}
        skip_write = existing_data == new_data

    if skip_write:
        print(f"Benchmark data unchanged, skipping write: {OUTPUT_FILE}")
    else:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"Built benchmark data: {OUTPUT_FILE}")

    print(f"  Datasets: {len(datasets)}")
    print(f"  Leaderboard entries: {len(leaderboard)}")
    print(f"  Total runs: {len(all_runs)}")
    for entry in leaderboard:
        prov = " (provisional)" if entry["provisional"] else ""
        print(f"  #{entry['rank']} {entry['model']} on {entry['gpu_name']}"
              f" — keep {entry['keep_rate_mean']}% | best bpb {entry['best_bpb']}"
              f" | {entry['runs']} runs{prov}")


if __name__ == "__main__":
    build_leaderboard()
