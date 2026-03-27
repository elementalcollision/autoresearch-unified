#!/usr/bin/env python3
"""Power and energy analysis for autoresearch experiment runs.

Analyzes completed experiment runs to produce:
1. Per-experiment energy summary (watts, joules/token, total energy)
2. Energy efficiency ranking (joules per unit of val_bpb improvement)
3. Cross-platform energy comparison (GPU-grouped)
4. Power vs. performance analysis (Pareto frontier)
5. Summary statistics (per-platform aggregates, kWh totals)

Usage:
    python scripts/power_analysis.py [--results-dir DIR] [--output-dir DIR]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Data loading (standalone, no tui dependency needed)
# ---------------------------------------------------------------------------

@dataclass
class Result:
    exp: str
    description: str
    val_bpb: float
    peak_mem_gb: float
    tok_sec: int
    mfu: float
    steps: int
    status: str
    notes: str
    gpu_name: str = ""
    baseline_sha: str = ""
    watts: float = 0.0
    joules_per_token: float = 0.0
    total_energy_joules: float = 0.0


def load_tsv(path: str) -> list[Result]:
    """Load results from a TSV file. Handles 10, 11, and 14-column formats."""
    results = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                results.append(Result(
                    exp=row["exp"],
                    description=row.get("description", ""),
                    val_bpb=float(row.get("val_bpb", 0) or 0),
                    peak_mem_gb=float(row.get("peak_mem_gb", 0) or 0),
                    tok_sec=int(float(row.get("tok_sec", 0) or 0)),
                    mfu=float(row.get("mfu", 0) or 0),
                    steps=int(float(row.get("steps", 0) or 0)),
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                    gpu_name=row.get("gpu_name", ""),
                    baseline_sha=row.get("baseline_sha", ""),
                    watts=float(row.get("watts", 0) or 0),
                    joules_per_token=float(row.get("joules_per_token", 0) or 0),
                    total_energy_joules=float(row.get("total_energy_joules", 0) or 0),
                ))
            except (ValueError, KeyError):
                continue
    return results


def load_all_results(results_dir: str) -> dict[str, list[Result]]:
    """Recursively find and load all TSV files under results_dir."""
    datasets: dict[str, list[Result]] = {}
    for root, _dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if fname.endswith(".tsv"):
                path = os.path.join(root, fname)
                results = load_tsv(path)
                if results:
                    rel = os.path.relpath(path, results_dir)
                    datasets[rel] = results
    return datasets


# ---------------------------------------------------------------------------
# Analysis 1: Per-Experiment Energy Summary
# ---------------------------------------------------------------------------

def energy_summary(results: list[Result], label: str) -> str:
    """Generate per-experiment energy summary table."""
    lines = [f"\n### Energy Summary: {label}\n"]
    lines.append(f"{'Exp':<6} {'GPU':<30} {'val_bpb':>8} {'Watts':>7} {'J/tok':>10} "
                 f"{'Total J':>10} {'Status':<8}")
    lines.append("-" * 95)

    has_power = False
    for r in results:
        flag = "" if r.watts > 0 else " *"
        if r.watts > 0:
            has_power = True
        lines.append(
            f"{r.exp:<6} {r.gpu_name[:30]:<30} {r.val_bpb:>8.4f} {r.watts:>7.1f} "
            f"{r.joules_per_token:>10.6f} {r.total_energy_joules:>10.1f} {r.status:<8}{flag}"
        )

    if not has_power:
        lines.append("\n* All experiments show 0.0W — power instrumentation was not active during these runs.")
    elif any(r.watts == 0 for r in results):
        lines.append("\n* Rows marked with * have no power data (instrumentation unavailable or crash).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 2: Energy Efficiency Ranking
# ---------------------------------------------------------------------------

def efficiency_ranking(results: list[Result]) -> str:
    """Rank kept experiments by energy efficiency."""
    lines = ["\n### Energy Efficiency Ranking\n"]

    # Find baseline
    baselines = [r for r in results if r.status == "baseline"]
    kept = [r for r in results if r.status == "keep" and r.watts > 0]

    if not baselines:
        lines.append("No baseline found — cannot compute efficiency ranking.")
        return "\n".join(lines)

    baseline_bpb = baselines[0].val_bpb

    if not kept:
        lines.append("No kept experiments with power data — cannot rank.")
        return "\n".join(lines)

    # Compute energy cost per unit of val_bpb improvement
    ranked = []
    for r in kept:
        bpb_improvement = baseline_bpb - r.val_bpb
        if bpb_improvement > 0:
            energy_per_improvement = r.joules_per_token / bpb_improvement
            ranked.append((r, bpb_improvement, energy_per_improvement))

    if not ranked:
        lines.append("No kept experiments improved over baseline — cannot rank.")
        return "\n".join(lines)

    ranked.sort(key=lambda x: x[2])  # lowest energy per improvement first

    lines.append(f"Baseline val_bpb: {baseline_bpb:.6f}\n")
    lines.append(f"{'Rank':<5} {'Exp':<6} {'val_bpb':>8} {'Improve':>8} {'J/tok':>10} "
                 f"{'J/tok/improve':>14}")
    lines.append("-" * 60)

    for i, (r, improve, cost) in enumerate(ranked, 1):
        lines.append(
            f"{i:<5} {r.exp:<6} {r.val_bpb:>8.4f} {improve:>8.6f} "
            f"{r.joules_per_token:>10.6f} {cost:>14.4f}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 3: Cross-Platform Energy Comparison
# ---------------------------------------------------------------------------

def cross_platform_comparison(all_results: dict[str, list[Result]]) -> str:
    """Compare energy metrics across GPU platforms."""
    lines = ["\n### Cross-Platform Energy Comparison\n"]

    # Group all results by gpu_name
    by_gpu: dict[str, list[Result]] = defaultdict(list)
    for results in all_results.values():
        for r in results:
            if r.gpu_name:
                by_gpu[r.gpu_name].append(r)

    if not by_gpu:
        lines.append("No GPU data available.")
        return "\n".join(lines)

    lines.append(f"{'GPU':<35} {'Exps':>5} {'Avg W':>7} {'Avg J/tok':>10} "
                 f"{'Avg MFU':>8} {'Avg tok/s':>10} {'W/tok_s':>8}")
    lines.append("-" * 95)

    for gpu_name in sorted(by_gpu.keys()):
        exps = by_gpu[gpu_name]
        # Only include non-crash experiments
        valid = [r for r in exps if r.status != "crash"]
        if not valid:
            continue

        avg_watts = np.mean([r.watts for r in valid])
        avg_jpt = np.mean([r.joules_per_token for r in valid]) if any(r.joules_per_token > 0 for r in valid) else 0.0
        mfu_vals = [r.mfu for r in valid if r.mfu > 0]
        avg_mfu = np.mean(mfu_vals) if mfu_vals else 0.0
        toks_vals = [r.tok_sec for r in valid if r.tok_sec > 0]
        avg_toks = np.mean(toks_vals) if toks_vals else 0.0
        # Power proportionality: watts per tok/s
        w_per_toks = avg_watts / avg_toks if avg_toks > 0 and avg_watts > 0 else 0.0

        lines.append(
            f"{gpu_name[:35]:<35} {len(valid):>5} {avg_watts:>7.1f} {avg_jpt:>10.6f} "
            f"{avg_mfu:>8.1f} {avg_toks:>10.0f} {w_per_toks:>8.4f}"
        )

    has_power = any(r.watts > 0 for results in all_results.values() for r in results)
    if not has_power:
        lines.append("\nNote: No power data available yet. Run experiments with power instrumentation enabled.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 4: Power vs. Performance
# ---------------------------------------------------------------------------

def power_performance_analysis(results: list[Result]) -> str:
    """Analyze power-performance relationships."""
    lines = ["\n### Power vs. Performance Analysis\n"]

    powered = [r for r in results if r.watts > 0 and r.status != "crash"]
    if not powered:
        lines.append("No experiments with power data — skipping power-performance analysis.")
        return "\n".join(lines)

    # 4a: Watts vs tok_sec (power proportionality)
    lines.append("**Power Proportionality (Watts vs. Throughput)**")
    lines.append(f"{'Exp':<6} {'Watts':>7} {'tok/s':>10} {'W per ktok/s':>13}")
    lines.append("-" * 40)
    for r in sorted(powered, key=lambda x: x.tok_sec, reverse=True):
        w_per_ktoks = (r.watts / (r.tok_sec / 1000)) if r.tok_sec > 0 else 0
        lines.append(f"{r.exp:<6} {r.watts:>7.1f} {r.tok_sec:>10} {w_per_ktoks:>13.2f}")

    # 4b: Pareto frontier — best val_bpb per joule
    lines.append("\n**Pareto Frontier: Best val_bpb per Joule**")
    # Sort by joules_per_token ascending, track best val_bpb seen
    by_energy = sorted(powered, key=lambda x: x.joules_per_token)
    pareto = []
    best_bpb = float("inf")
    for r in by_energy:
        if r.val_bpb > 0 and r.val_bpb < best_bpb:
            best_bpb = r.val_bpb
            pareto.append(r)

    if pareto:
        lines.append(f"{'Exp':<6} {'val_bpb':>8} {'J/tok':>10} {'Status':<8}")
        lines.append("-" * 40)
        for r in pareto:
            lines.append(f"{r.exp:<6} {r.val_bpb:>8.4f} {r.joules_per_token:>10.6f} {r.status:<8}")
    else:
        lines.append("Insufficient data for Pareto analysis.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analysis 5: Summary Statistics
# ---------------------------------------------------------------------------

def summary_statistics(all_results: dict[str, list[Result]]) -> str:
    """Compute per-platform summary statistics."""
    lines = ["\n### Power Summary Statistics\n"]

    by_gpu: dict[str, list[Result]] = defaultdict(list)
    for results in all_results.values():
        for r in results:
            if r.gpu_name and r.status != "crash":
                by_gpu[r.gpu_name].append(r)

    if not by_gpu:
        lines.append("No data available.")
        return "\n".join(lines)

    total_energy_all = 0.0

    for gpu_name in sorted(by_gpu.keys()):
        exps = by_gpu[gpu_name]
        watts_vals = [r.watts for r in exps if r.watts > 0]
        jpt_vals = [r.joules_per_token for r in exps if r.joules_per_token > 0]
        total_energy = sum(r.total_energy_joules for r in exps)
        total_energy_all += total_energy

        lines.append(f"**{gpu_name}** ({len(exps)} experiments)")

        if watts_vals:
            w = np.array(watts_vals)
            lines.append(f"  Power draw:    mean={np.mean(w):.1f}W  std={np.std(w):.1f}W  "
                         f"min={np.min(w):.1f}W  max={np.max(w):.1f}W  "
                         f"CV={np.std(w)/np.mean(w)*100:.1f}%")
        else:
            lines.append("  Power draw:    no instrumented data")

        if jpt_vals:
            j = np.array(jpt_vals)
            lines.append(f"  Joules/token:  mean={np.mean(j):.6f}  std={np.std(j):.6f}  "
                         f"min={np.min(j):.6f}  max={np.max(j):.6f}")
        else:
            lines.append("  Joules/token:  no instrumented data")

        kwh = total_energy / 3_600_000  # joules to kWh
        lines.append(f"  Total energy:  {total_energy:.1f} J  ({kwh:.4f} kWh)")
        lines.append("")

    total_kwh = total_energy_all / 3_600_000
    lines.append(f"**Total energy across all platforms: {total_energy_all:.1f} J ({total_kwh:.4f} kWh)**")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(all_results: dict[str, list[Result]], output_dir: str) -> None:
    """Generate full power analysis report."""
    os.makedirs(output_dir, exist_ok=True)

    sections = []
    sections.append("# Power & Energy Analysis Report\n")
    sections.append("Analysis of accelerator power consumption across autoresearch experiments.\n")

    # Determine if any power data exists
    has_power = any(
        r.watts > 0
        for results in all_results.values()
        for r in results
    )

    if not has_power:
        sections.append("> **Note**: No power-instrumented data found. All energy values are 0.0.\n"
                        "> Run experiments with the power instrumentation (merged in PR #24) to populate these metrics.\n")

    # Per-file summaries
    for label, results in sorted(all_results.items()):
        sections.append(energy_summary(results, label))
        sections.append(efficiency_ranking(results))
        sections.append(power_performance_analysis(results))

    # Cross-platform (all data combined)
    sections.append(cross_platform_comparison(all_results))
    sections.append(summary_statistics(all_results))

    report = "\n\n".join(sections) + "\n"

    # Print to stdout
    print(report)

    # Write markdown report
    report_path = os.path.join(output_dir, "power-analysis.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to: {report_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze power and energy consumption across autoresearch experiments.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing TSV result files (default: results/)",
    )
    parser.add_argument(
        "--output-dir",
        default="docs",
        help="Directory for output reports (default: docs/)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    all_results = load_all_results(args.results_dir)
    if not all_results:
        print(f"No TSV files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v) for v in all_results.values())} experiments "
          f"from {len(all_results)} files", file=sys.stderr)

    generate_report(all_results, args.output_dir)


if __name__ == "__main__":
    main()
