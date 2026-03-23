#!/usr/bin/env python3
"""
Lightweight experiment progress monitor.

Reads results.tsv files and prints a live summary. Runs in a separate
terminal with zero interaction with the training process — just reads
files on disk.

Usage:
    # Auto-refresh every 30s (default)
    python3 monitor.py

    # Custom refresh interval
    python3 monitor.py --interval 10

    # One-shot (no refresh)
    python3 monitor.py --once
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"

DATASET_ORDER = ["climbmix", "fineweb-edu", "cosmopedia-v2", "slimpajama", "fineweb-edu-high", "fineweb", "github-code-python"]


def parse_results(tsv_path):
    """Parse a results.tsv file into experiment records."""
    if not tsv_path.exists():
        return []
    records = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("exp\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 8:
                try:
                    val_bpb = float(parts[2])
                except ValueError:
                    val_bpb = 0.0
                records.append({
                    "exp": parts[0],
                    "description": parts[1][:55],
                    "val_bpb": val_bpb,
                    "mem_gb": parts[3],
                    "tok_sec": parts[4],
                    "mfu": parts[5],
                    "steps": parts[6],
                    "status": parts[7],
                })
    return records


def dataset_summary(name, records):
    """Generate summary stats for a dataset."""
    total = len(records)
    keeps = [r for r in records if r["status"] == "keep"]
    discards = [r for r in records if r["status"] == "discard"]
    crashes = [r for r in records if r["status"] == "crash"]
    baselines = [r for r in records if r["status"] == "baseline"]

    best_val = None
    baseline_val = None
    for r in records:
        if r["val_bpb"] > 0:
            if r["status"] in ("keep", "baseline"):
                if best_val is None or r["val_bpb"] < best_val:
                    best_val = r["val_bpb"]
            if r["status"] == "baseline":
                baseline_val = r["val_bpb"]

    # If baseline crashed, use first keep as reference
    if baseline_val is None or baseline_val == 0:
        for r in records:
            if r["status"] == "keep" and r["val_bpb"] > 0:
                baseline_val = r["val_bpb"]
                break

    improvement = ""
    if best_val and baseline_val and baseline_val > 0:
        delta = baseline_val - best_val
        pct = 100 * delta / baseline_val
        improvement = f"-{pct:.1f}%"

    return {
        "name": name,
        "total": total,
        "keeps": len(keeps),
        "discards": len(discards),
        "crashes": len(crashes),
        "baseline": baseline_val,
        "best": best_val,
        "improvement": improvement,
        "keep_rate": f"{100*len(keeps)/total:.0f}%" if total > 0 else "—",
        "last_exp": records[-1] if records else None,
    }


def render(summaries):
    """Render the monitor display."""
    now = datetime.now().strftime("%H:%M:%S")
    lines = []
    lines.append("")
    lines.append(f"  ╔══════════════════════════════════════════════════════════════════╗")
    lines.append(f"  ║  Autoresearch Experiment Monitor              {now}        ║")
    lines.append(f"  ╠══════════════════════════════════════════════════════════════════╣")

    active = [s for s in summaries if s["total"] > 0]
    pending = [s for s in summaries if s["total"] == 0]

    if not active:
        lines.append(f"  ║  No experiments found. Waiting for results...                  ║")
        lines.append(f"  ╚══════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    for s in active:
        best_str = f"{s['best']:.6f}" if s['best'] else "—"
        base_str = f"{s['baseline']:.4f}" if s['baseline'] else "crash"
        imp_str = s['improvement'] or "—"

        # Progress bar (assuming 100 max)
        pct = min(s['total'] / 100, 1.0)
        bar_len = 20
        filled = int(pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        lines.append(f"  ║                                                                  ║")
        lines.append(f"  ║  {s['name'].upper():<18} {bar} {s['total']:>3}/100                    ║")
        lines.append(f"  ║    Best: {best_str:<12} Baseline: {base_str:<10} Improvement: {imp_str:<8}║")
        lines.append(f"  ║    Keeps: {s['keeps']:<4} Discards: {s['discards']:<4} Crashes: {s['crashes']:<4} Rate: {s['keep_rate']:<6}  ║")

        if s["last_exp"]:
            last = s["last_exp"]
            status_icon = {"keep": "✓", "discard": "✗", "crash": "💥", "baseline": "◆"}.get(last["status"], "?")
            desc = last["description"][:45]
            lines.append(f"  ║    Latest: {status_icon} {last['exp']:<6} {last['val_bpb']:.4f}  {desc:<45}║")

        lines.append(f"  ║                                                                  ║")
        lines.append(f"  ╠══════════════════════════════════════════════════════════════════╣")

    # Kept experiments detail for active datasets
    for s_info in active:
        name = s_info["name"]
        tsv_path = RESULTS_DIR / name / "results.tsv"
        records = parse_results(tsv_path)
        keeps = [r for r in records if r["status"] == "keep"]
        if keeps:
            lines.append(f"  ║  {name} keeps:                                                   ║")
            for k in keeps[-5:]:  # Last 5 keeps
                desc = k["description"][:40]
                lines.append(f"  ║    {k['exp']:<7} {k['val_bpb']:.6f}  {desc:<40}        ║")
            if len(keeps) > 5:
                lines.append(f"  ║    ... and {len(keeps)-5} earlier keeps                                    ║")
            lines.append(f"  ╠══════════════════════════════════════════════════════════════════╣")

    # Pending datasets
    if pending:
        pending_names = ", ".join(s["name"] for s in pending)
        lines.append(f"  ║  Pending: {pending_names:<56}║")
        lines.append(f"  ╠══════════════════════════════════════════════════════════════════╣")

    lines[-1] = f"  ╚══════════════════════════════════════════════════════════════════╝"
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Lightweight experiment progress monitor")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval in seconds (default: 30)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    try:
        while True:
            summaries = []
            for name in DATASET_ORDER:
                tsv_path = RESULTS_DIR / name / "results.tsv"
                records = parse_results(tsv_path)
                summaries.append(dataset_summary(name, records))

            # Clear screen
            if not args.once:
                print("\033[2J\033[H", end="")

            print(render(summaries))

            if not args.once:
                print(f"\n  Refreshing every {args.interval}s. Ctrl+C to stop.")
                time.sleep(args.interval)
            else:
                break

    except KeyboardInterrupt:
        print("\n  Monitor stopped.")


if __name__ == "__main__":
    main()
