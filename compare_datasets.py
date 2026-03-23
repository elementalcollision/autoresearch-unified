#!/usr/bin/env python3
"""
Cross-dataset comparison and analysis tool.

Reads results from results/<dataset>/results.tsv for each completed dataset
and generates comparative visualizations and analysis.

Usage:
    uv run compare_datasets.py                    # Full comparison
    uv run compare_datasets.py --output chart.png  # Custom output path
    uv run compare_datasets.py --summary           # Text summary only
    uv run compare_datasets.py --wiki              # Generate + push to wiki
"""

import os
import re
import sys
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"

# Dark theme palette (matches CUDA repo style)
DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
GRID_COLOR = "#2a2a4a"
TEXT_COLOR = "#e0e0e0"
TITLE_COLOR = "#ffffff"
ACCENT_GREEN = "#2ecc71"
ACCENT_RED = "#e74c3c"
ACCENT_ORANGE = "#f39c12"
ACCENT_BLUE = "#3498db"

DATASET_COLORS = {
    "climbmix": "#e74c3c",
    "fineweb-edu": "#2ecc71",
    "cosmopedia-v2": "#3498db",
    "slimpajama": "#9b59b6",
    "openwebtext": "#f39c12",
    "pubmed-abstract": "#1abc9c",
}

# Hyperparameters we track across datasets
TRACKED_PARAMS = [
    "MATRIX_LR", "SCALAR_LR", "EMBEDDING_LR", "UNEMBEDDING_LR",
    "WEIGHT_DECAY", "WARMDOWN_RATIO", "FINAL_LR_FRAC",
]

# Known baseline values for ratio computation
BASELINE_DEFAULTS = {
    "MATRIX_LR": 0.04,
    "SCALAR_LR": 0.5,
    "EMBEDDING_LR": 0.6,
    "UNEMBEDDING_LR": 0.004,
    "WEIGHT_DECAY": 0.2,
    "WARMDOWN_RATIO": 0.5,
    "FINAL_LR_FRAC": 0.0,
}


@dataclass
class Experiment:
    name: str
    description: str
    val_bpb: float
    peak_mem_gb: float
    tok_sec: float
    mfu: float
    steps: int
    status: str
    notes: str = ""


@dataclass
class DatasetRun:
    dataset: str
    experiments: list = field(default_factory=list)

    @property
    def baseline(self):
        for e in self.experiments:
            if e.status == "baseline":
                return e
        return self.experiments[0] if self.experiments else None

    @property
    def best(self):
        valid = [e for e in self.experiments if e.val_bpb > 0.5]
        return min(valid, key=lambda e: e.val_bpb) if valid else None

    @property
    def kept(self):
        return [e for e in self.experiments if e.status in ("keep", "kept")]

    @property
    def discarded(self):
        return [e for e in self.experiments if e.status == "discard"]

    @property
    def crashed(self):
        return [e for e in self.experiments if e.status == "crash"]

    @property
    def improvement_pct(self):
        b = self.baseline
        best = self.best
        if b and best and b.val_bpb > 0:
            return (b.val_bpb - best.val_bpb) / b.val_bpb * 100
        return 0.0

    @property
    def keep_rate(self):
        total = len(self.experiments)
        return len(self.kept) / total * 100 if total > 0 else 0.0

    @property
    def baseline_depth(self):
        """Extract depth from baseline notes (e.g. 'depth=12, AMD ...')."""
        bl = self.baseline
        if bl and bl.notes:
            m = re.search(r"depth=(\d+)", bl.notes)
            if m:
                return int(m.group(1))
        return None

    @property
    def best_depth(self):
        """Track depth changes through the keep chain."""
        depth = self.baseline_depth
        for e in self.kept:
            desc = e.description.lower()
            if "depth" in desc:
                m = re.search(r"to\s+(\d+)", desc)
                if m:
                    depth = int(m.group(1))
        return depth

    @property
    def optimal_config(self):
        """Extract the optimal hyperparameter values from the best experiment."""
        changes = {}
        # Use best experiment + any kept chain; if no kept, just best
        chain = self.kept if self.kept else ([self.best] if self.best else [])
        for e in chain:
            desc = e.description
            desc_lower = desc.lower()
            for param in TRACKED_PARAMS + ["depth", "window_pattern",
                                            "device_batch_size", "total_batch_size",
                                            "head_dim", "aspect_ratio"]:
                key = param.lower().replace("_", "")
                if key not in desc_lower.replace("_", "").replace(" ", ""):
                    continue

                # Try "from X to Y" pattern — extract Y
                m = re.search(r'from\s+[\w."]+\s+to\s+([\w."]+)', desc_lower)
                if m:
                    val = m.group(1).strip().strip('".')
                    changes[param.upper()] = val
                    continue

                # Try "Change X to Y" or quoted values
                m = re.search(r'to\s+"([^"]+)"', desc_lower)
                if m:
                    changes[param.upper()] = m.group(1)
                    continue

                # Fallback: last numeric token after "to"
                m = re.search(r'\bto\s+([\d.]+)', desc_lower)
                if m:
                    changes[param.upper()] = m.group(1)
        return changes

    def get_optimal_value(self, param):
        """Get the final tuned value for a specific parameter."""
        config = self.optimal_config
        val_str = config.get(param)
        if val_str:
            try:
                return float(val_str)
            except ValueError:
                return None
        return None


def load_results(dataset_name):
    """Load results.tsv for a dataset."""
    tsv_path = RESULTS_DIR / dataset_name / "results.tsv"
    if not tsv_path.exists():
        return None

    run = DatasetRun(dataset=dataset_name)
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("exp\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            try:
                exp = Experiment(
                    name=parts[0],
                    description=parts[1],
                    val_bpb=float(parts[2]),
                    peak_mem_gb=float(parts[3]),
                    tok_sec=float(parts[4]),
                    mfu=float(parts[5]),
                    steps=int(parts[6]),
                    status=parts[7],
                    notes=parts[8] if len(parts) > 8 else "",
                )
                run.experiments.append(exp)
            except (ValueError, IndexError):
                continue

    return run if run.experiments else None


def load_all_results():
    """Load results for all datasets that have them."""
    runs = {}
    if not RESULTS_DIR.exists():
        return runs

    for d in sorted(RESULTS_DIR.iterdir()):
        if d.is_dir():
            run = load_results(d.name)
            if run:
                runs[d.name] = run
    return runs


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def print_summary(runs):
    """Print a text comparison summary."""
    if not runs:
        print("No results found. Run experiments first.")
        return

    print("\n" + "=" * 80)
    print("  CROSS-DATASET COMPARISON")
    print("=" * 80)

    # Overview table
    print(f"\n  {'Dataset':<20} {'Exps':>5} {'Kept':>5} {'Crash':>5} {'Baseline':>10} {'Best':>10} {'Improv':>8} {'Mem GB':>8}")
    print("  " + "-" * 73)

    for name, run in runs.items():
        bl = run.baseline
        best = run.best
        print(f"  {name:<20} {len(run.experiments):>5} {len(run.kept):>5} {len(run.crashed):>5} "
              f"{bl.val_bpb:>10.6f} {best.val_bpb:>10.6f} {run.improvement_pct:>7.1f}% {best.peak_mem_gb:>7.1f}")

    # Optimal configs comparison
    print(f"\n  Optimal Hyperparameter Comparison")
    print("  " + "-" * 73)

    all_params = set()
    for run in runs.values():
        all_params.update(run.optimal_config.keys())

    if all_params:
        print(f"  {'Parameter':<25}", end="")
        for name in runs:
            print(f" {name:<15}", end="")
        print()

        for param in sorted(all_params):
            print(f"  {param:<25}", end="")
            for name, run in runs.items():
                val = run.optimal_config.get(param, "default")
                print(f" {val:<15}", end="")
            print()

    # Key insights
    print(f"\n  Key Observations")
    print("  " + "-" * 73)

    if len(runs) >= 2:
        best_improv = max(runs.items(), key=lambda x: x[1].improvement_pct)
        print(f"  - Biggest improvement: {best_improv[0]} ({best_improv[1].improvement_pct:.1f}%)")

        best_abs = min(runs.items(), key=lambda x: x[1].best.val_bpb)
        print(f"  - Best absolute val_bpb: {best_abs[0]} ({best_abs[1].best.val_bpb:.6f})")

        best_keep = max(runs.items(), key=lambda x: len(x[1].kept) / max(len(x[1].experiments), 1))
        keep_rate = len(best_keep[1].kept) / len(best_keep[1].experiments) * 100
        print(f"  - Highest keep rate: {best_keep[0]} ({keep_rate:.0f}%)")

        configs = {name: run.optimal_config for name, run in runs.items()}
        shared_params = set.intersection(*[set(c.keys()) for c in configs.values()]) if configs else set()
        differing = [p for p in shared_params if len(set(configs[n].get(p, "default") for n in runs)) > 1]

        if differing:
            print(f"  - Parameters that differ across datasets: {', '.join(sorted(differing))}")
        else:
            print(f"  - All tuned parameters agree across datasets (hyperparameters transfer!)")

    print()


# ---------------------------------------------------------------------------
# Dark-themed visualization (matches CUDA repo style)
# ---------------------------------------------------------------------------

def _get_color(name):
    """Get color for a dataset, with fallback."""
    return DATASET_COLORS.get(name, "#95a5a6")


def _style_axis(ax, title):
    """Apply dark theme styling to an axis."""
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def generate_chart(runs, output_path):
    """Generate a dark-themed 6-panel comparison chart matching CUDA repo style."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed. Run: uv pip install matplotlib")
        return

    n_datasets = len(runs)
    if n_datasets == 0:
        print("No results to chart.")
        return

    names = list(runs.keys())
    x = list(range(len(names)))

    # Create figure with dark background
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Cross-Dataset Comparison \u2014 AMD Instinct MI300X (192 GB)",
                 fontsize=18, fontweight="bold", color=TITLE_COLOR, y=0.98)

    # -----------------------------------------------------------------------
    # Panel 1: Baseline vs Best val_bpb
    # -----------------------------------------------------------------------
    ax1 = axes[0, 0]
    _style_axis(ax1, "Baseline \u2192 Best val_bpb")

    baselines = [runs[n].baseline.val_bpb for n in names]
    bests = [runs[n].best.val_bpb for n in names]
    bar_w = 0.35

    ax1.bar([i - bar_w / 2 for i in x], baselines, bar_w,
            label="Baseline", color="#555555", edgecolor="#777777", linewidth=0.5)
    ax1.bar([i + bar_w / 2 for i in x], bests, bar_w,
            label="Best", color=[_get_color(n) for n in names],
            edgecolor="#ffffff", linewidth=0.5)

    for i, (bl, best) in enumerate(zip(baselines, bests)):
        improv = (bl - best) / bl * 100
        ax1.annotate(f"-{improv:.1f}%", xy=(i + bar_w / 2, best),
                     xytext=(0, -16), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold",
                     color=ACCENT_GREEN)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax1.set_ylabel("val_bpb")
    legend = ax1.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                        labelcolor=TEXT_COLOR)

    # -----------------------------------------------------------------------
    # Panel 2: Experiment Outcomes (stacked bar)
    # -----------------------------------------------------------------------
    ax2 = axes[0, 1]
    _style_axis(ax2, "Experiment Outcomes")

    keeps = [len(runs[n].kept) for n in names]
    discards = [len(runs[n].discarded) for n in names]
    crashes = [len(runs[n].crashed) for n in names]

    bars_k = ax2.bar(x, keeps, 0.6, label="Keep", color=ACCENT_GREEN,
                     edgecolor="#ffffff", linewidth=0.5)
    bars_d = ax2.bar(x, discards, 0.6, bottom=keeps, label="Discard",
                     color=ACCENT_RED, edgecolor="#ffffff", linewidth=0.5)
    bars_c = ax2.bar(x, crashes, 0.6,
                     bottom=[k + d for k, d in zip(keeps, discards)],
                     label="Crash", color=ACCENT_ORANGE,
                     edgecolor="#ffffff", linewidth=0.5)

    # Label counts on each segment
    for i, (k, d, c) in enumerate(zip(keeps, discards, crashes)):
        if k > 0:
            ax2.text(i, k / 2, str(k), ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
        if d > 0:
            ax2.text(i, k + d / 2, str(d), ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")
        if c > 0:
            ax2.text(i, k + d + c / 2, str(c), ha="center", va="center",
                     fontsize=9, fontweight="bold", color="white")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Experiments")
    ax2.legend(fontsize=8, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR)

    # -----------------------------------------------------------------------
    # Panel 3: Resource Usage (memory + throughput)
    # -----------------------------------------------------------------------
    ax3 = axes[0, 2]
    _style_axis(ax3, "Resource Usage")

    mem_baseline = [runs[n].baseline.peak_mem_gb for n in names]
    mem_best = [runs[n].best.peak_mem_gb for n in names]
    tok_baseline = [runs[n].baseline.tok_sec for n in names]
    tok_best = [runs[n].best.tok_sec for n in names]

    bar_w2 = 0.35
    bars_mem_bl = ax3.bar([i - bar_w2 / 2 for i in x], mem_baseline, bar_w2,
                          label="Mem baseline", color="#555555",
                          edgecolor="#777777", linewidth=0.5)
    bars_mem_best = ax3.bar([i + bar_w2 / 2 for i in x], mem_best, bar_w2,
                            label="Mem best", color=[_get_color(n) for n in names],
                            edgecolor="#ffffff", linewidth=0.5, alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax3.set_ylabel("Peak Memory (GB)", color=TEXT_COLOR)
    ax3.legend(fontsize=7, loc="upper left", facecolor=PANEL_BG,
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Secondary y-axis for throughput
    ax3b = ax3.twinx()
    ax3b.plot(x, tok_baseline, "s--", color="#888888", markersize=6,
              label="tok/s baseline", alpha=0.7)
    ax3b.plot(x, tok_best, "D-", color=ACCENT_ORANGE, markersize=6,
              label="tok/s best", linewidth=2)
    ax3b.set_ylabel("Throughput (tok/s)", color=ACCENT_ORANGE)
    ax3b.tick_params(axis="y", colors=ACCENT_ORANGE, labelsize=9)
    ax3b.spines["right"].set_color(ACCENT_ORANGE)
    ax3b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    ax3b.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG,
                edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # -----------------------------------------------------------------------
    # Panel 4: Hyperparameter Ratios (normalized to baseline = 1.0)
    # -----------------------------------------------------------------------
    ax4 = axes[1, 0]
    _style_axis(ax4, "Hyperparameter Ratios (1.0 = default)")

    # Only show params that have data in at least one dataset
    active_params = []
    for p in TRACKED_PARAMS:
        for run in runs.values():
            if run.get_optimal_value(p) is not None:
                active_params.append(p)
                break

    if active_params:
        n_params = len(active_params)
        total_w = 0.7
        w = total_w / n_datasets
        param_x = list(range(n_params))

        for di, name in enumerate(names):
            run = runs[name]
            ratios = []
            for p in active_params:
                opt = run.get_optimal_value(p)
                default = BASELINE_DEFAULTS.get(p, 1.0)
                if opt is not None and default != 0:
                    ratios.append(opt / default)
                else:
                    ratios.append(1.0)

            offset = -total_w / 2 + w * di + w / 2
            ax4.bar([px + offset for px in param_x], ratios, w,
                    label=name, color=_get_color(name),
                    edgecolor="#ffffff", linewidth=0.5, alpha=0.85)

        ax4.axhline(y=1.0, color=TEXT_COLOR, linestyle="--", alpha=0.5, linewidth=1)
        ax4.set_xticks(param_x)
        # Shorten param names for display
        short_names = [p.replace("_LR", "").replace("EMBEDDING", "Embed")
                       .replace("UNEMBEDDING", "Unembed").replace("MATRIX", "Matrix")
                       .replace("SCALAR", "Scalar").replace("WEIGHT_DECAY", "WD")
                       .replace("WARMDOWN_RATIO", "Warmdown").replace("FINAL_LR_FRAC", "FinalLR")
                       for p in active_params]
        ax4.set_xticklabels(short_names, rotation=35, ha="right", fontsize=8)
        ax4.set_ylabel("Ratio vs default")
        ax4.legend(fontsize=7, facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                   labelcolor=TEXT_COLOR)
    else:
        ax4.text(0.5, 0.5, "No hyperparameter data", transform=ax4.transAxes,
                 ha="center", va="center", color=TEXT_COLOR, fontsize=12)

    # -----------------------------------------------------------------------
    # Panel 5: Architecture Convergence (depth + steps)
    # -----------------------------------------------------------------------
    ax5 = axes[1, 1]
    _style_axis(ax5, "Architecture Convergence")

    bl_depths = []
    best_depths = []
    bl_steps = []
    best_steps = []
    for n in names:
        run = runs[n]
        bl_depths.append(run.baseline_depth or 0)
        best_depths.append(run.best_depth or 0)
        bl_steps.append(run.baseline.steps if run.baseline else 0)
        best_steps.append(run.best.steps if run.best else 0)

    bar_w3 = 0.35
    ax5.bar([i - bar_w3 / 2 for i in x], bl_depths, bar_w3,
            label="Depth baseline", color="#555555", edgecolor="#777777", linewidth=0.5)
    ax5.bar([i + bar_w3 / 2 for i in x], best_depths, bar_w3,
            label="Depth best", color=[_get_color(n) for n in names],
            edgecolor="#ffffff", linewidth=0.5)

    # Label depth values
    for i, (bd, bst) in enumerate(zip(bl_depths, best_depths)):
        ax5.text(i - bar_w3 / 2, bd + 0.2, str(bd), ha="center",
                 fontsize=9, color=TEXT_COLOR, fontweight="bold")
        ax5.text(i + bar_w3 / 2, bst + 0.2, str(bst), ha="center",
                 fontsize=9, color=TITLE_COLOR, fontweight="bold")

    ax5.set_xticks(x)
    ax5.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax5.set_ylabel("Depth (layers)")
    ax5.legend(fontsize=7, loc="upper left", facecolor=PANEL_BG,
               edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # Secondary axis for steps
    ax5b = ax5.twinx()
    ax5b.plot(x, bl_steps, "s--", color="#888888", markersize=6,
              label="Steps baseline", alpha=0.7)
    ax5b.plot(x, best_steps, "D-", color=ACCENT_ORANGE, markersize=6,
              label="Steps best", linewidth=2)
    ax5b.set_ylabel("Steps (5 min budget)", color=ACCENT_ORANGE)
    ax5b.tick_params(axis="y", colors=ACCENT_ORANGE, labelsize=9)
    ax5b.spines["right"].set_color(ACCENT_ORANGE)
    ax5b.legend(fontsize=7, loc="upper right", facecolor=PANEL_BG,
                edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # -----------------------------------------------------------------------
    # Panel 6: Improvement & Keep Rate
    # -----------------------------------------------------------------------
    ax6 = axes[1, 2]
    _style_axis(ax6, "Improvement & Keep Rate")

    improvements = [runs[n].improvement_pct for n in names]
    keep_rates = [runs[n].keep_rate for n in names]

    bars_imp = ax6.bar(x, improvements, 0.6,
                       color=[_get_color(n) for n in names],
                       edgecolor="#ffffff", linewidth=0.5, alpha=0.85)

    # Label improvement bars
    for i, imp in enumerate(improvements):
        ax6.text(i, imp + 0.1, f"{imp:.1f}%", ha="center",
                 fontsize=10, fontweight="bold", color=TITLE_COLOR)

    ax6.set_xticks(x)
    ax6.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax6.set_ylabel("Improvement (%)")

    # Keep rate overlay line
    ax6b = ax6.twinx()
    ax6b.plot(x, keep_rates, "o-", color=ACCENT_ORANGE, markersize=8,
              linewidth=2.5, label="Keep Rate %", zorder=5)
    for i, kr in enumerate(zip(keep_rates)):
        ax6b.annotate(f"{kr[0]:.0f}%", xy=(i, kr[0]),
                      xytext=(8, 5), textcoords="offset points",
                      fontsize=9, color=ACCENT_ORANGE, fontweight="bold")

    ax6b.set_ylabel("Keep Rate (%)", color=ACCENT_ORANGE)
    ax6b.tick_params(axis="y", colors=ACCENT_ORANGE, labelsize=9)
    ax6b.spines["right"].set_color(ACCENT_ORANGE)
    ax6b.set_ylim(0, max(keep_rates) * 1.4 if keep_rates else 50)

    # Combined legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=ACCENT_ORANGE, label="Keep Rate %",
               markersize=6, linewidth=2),
    ]
    ax6b.legend(handles=legend_elements, fontsize=8, loc="upper right",
                facecolor=PANEL_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)

    # -----------------------------------------------------------------------
    # Finalize
    # -----------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Chart saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Wiki upload
# ---------------------------------------------------------------------------

def generate_wiki_analysis(runs):
    """Generate the full Cross-Dataset-Comparison wiki page markdown."""
    img_url = "https://raw.githubusercontent.com/wiki/elementalcollision/autoresearch-rocm/images/cross-dataset-comparison.png"
    lines = []
    lines.append("# Cross-Dataset Comparison")
    lines.append("")
    lines.append(f"[![Cross-Dataset Comparison]({img_url})]({img_url})")
    lines.append("")
    lines.append("_Click the chart to view full size._")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Experiments | Baseline | Best | Improvement | Peak tok/s |")
    lines.append("|---------|------------|----------|------|-------------|------------|")

    for name, run in runs.items():
        bl = run.baseline
        best = run.best
        imp = run.improvement_pct
        valid = [e for e in run.experiments if e.val_bpb > 0.5 and e.tok_sec > 0]
        peak_tok = max(e.tok_sec for e in valid) if valid else 0
        lines.append(
            f"| {name} | {len(run.experiments)} | {bl.val_bpb:.6f} | "
            f"{best.val_bpb:.6f} | {imp:+.2f}% | {peak_tok:,.0f} |"
        )

    lines.append("")
    lines.append("## Per-Dataset Analysis")
    lines.append("")

    for name, run in runs.items():
        bl = run.baseline
        best = run.best
        imp = run.improvement_pct
        valid = sorted(
            [e for e in run.experiments if e.val_bpb > 0.5],
            key=lambda e: e.val_bpb,
        )
        crashed = [e for e in run.experiments if e.val_bpb <= 0.5 or e.status == "crash"]
        tok_vals = [e.tok_sec for e in valid if e.tok_sec > 0]

        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- **Experiments:** {len(run.experiments)} total, {len(crashed)} crashed")
        lines.append(f"- **Baseline:** {bl.val_bpb:.6f} val_bpb ({bl.name})")
        lines.append(f"- **Best:** {best.val_bpb:.6f} val_bpb ({best.name}) — {best.description}")
        lines.append(f"- **Improvement:** {imp:+.3f}% over baseline")
        if tok_vals:
            lines.append(f"- **Throughput:** {min(tok_vals):,.0f} – {max(tok_vals):,.0f} tok/s "
                         f"(avg {sum(tok_vals)/len(tok_vals):,.0f})")
        lines.append("")

        # Top 5 table
        lines.append("| Rank | Experiment | val_bpb | tok/s | Description |")
        lines.append("|------|-----------|---------|-------|-------------|")
        for i, e in enumerate(valid[:5], 1):
            lines.append(f"| {i} | {e.name} | {e.val_bpb:.6f} | {e.tok_sec:,.0f} | {e.description[:60]} |")
        lines.append("")

    # Cross-dataset insights
    lines.append("## Key Insights")
    lines.append("")

    if len(runs) >= 2:
        best_improv = max(runs.items(), key=lambda x: x[1].improvement_pct)
        best_abs = min(runs.items(), key=lambda x: x[1].best.val_bpb)
        lines.append(f"- **Biggest improvement:** {best_improv[0]} ({best_improv[1].improvement_pct:+.2f}%)")
        lines.append(f"- **Best absolute val_bpb:** {best_abs[0]} ({best_abs[1].best.val_bpb:.6f})")

        # Check if any datasets converged on similar values
        best_bpbs = {name: run.best.val_bpb for name, run in runs.items()}
        vals = list(best_bpbs.values())
        if max(vals) - min(vals) < 0.005:
            lines.append("- **Convergence:** All datasets converged to similar optimal val_bpb "
                         f"(range: {min(vals):.6f} – {max(vals):.6f})")

        # Dataset-specific notes
        for name, run in runs.items():
            if run.improvement_pct < 0.01:
                lines.append(f"- **{name}:** Baseline hyperparameters are already optimal — "
                             "no experiment improved over default configuration")

    lines.append("")
    lines.append("## Hardware")
    lines.append("")
    lines.append("All experiments ran on **AMD Instinct MI300X** (192 GB HBM3) via RunPod.")
    lines.append("Training uses ROCm 6.x with CK Flash Attention and bf16 autocast.")
    lines.append("")
    lines.append("---")
    lines.append("_Generated by `compare_datasets.py --wiki`_")
    lines.append("")

    return "\n".join(lines)


def push_to_wiki(image_path, runs=None):
    """Clone wiki repo, add image, commit and push."""
    wiki_url = "https://github.com/elementalcollision/autoresearch-rocm.wiki.git"
    tmp_dir = tempfile.mkdtemp(prefix="rocm-wiki-")

    try:
        print(f"  Cloning wiki repo to {tmp_dir}...")
        subprocess.run(["git", "clone", wiki_url, tmp_dir],
                        check=True, capture_output=True, text=True)

        images_dir = Path(tmp_dir) / "images"
        images_dir.mkdir(exist_ok=True)

        dest = images_dir / "cross-dataset-comparison.png"
        shutil.copy2(image_path, dest)
        print(f"  Copied {image_path} -> {dest}")

        subprocess.run(["git", "add", "images/cross-dataset-comparison.png"],
                        cwd=tmp_dir, check=True, capture_output=True, text=True)

        # Write full wiki page with analysis if runs provided
        wiki_page = Path(tmp_dir) / "Cross-Dataset-Comparison.md"
        if runs:
            content = generate_wiki_analysis(runs)
            wiki_page.write_text(content)
            subprocess.run(["git", "add", "Cross-Dataset-Comparison.md"],
                            cwd=tmp_dir, check=True, capture_output=True, text=True)
            print("  Generated full wiki analysis page")
        elif wiki_page.exists():
            # Fallback: just ensure clickable image ref
            content = wiki_page.read_text()
            img_url = "https://raw.githubusercontent.com/wiki/elementalcollision/autoresearch-rocm/images/cross-dataset-comparison.png"
            clickable_img = f"[![Cross-Dataset Comparison]({img_url})]({img_url})"
            old_img_ref = f"![Cross-Dataset Comparison]({img_url})"
            if clickable_img not in content:
                content = content.replace(old_img_ref, clickable_img) if old_img_ref in content else content
                wiki_page.write_text(content)
                subprocess.run(["git", "add", "Cross-Dataset-Comparison.md"],
                                cwd=tmp_dir, check=True, capture_output=True, text=True)

        result = subprocess.run(["git", "diff", "--cached", "--quiet"],
                                cwd=tmp_dir, capture_output=True)
        if result.returncode != 0:
            subprocess.run(
                ["git", "commit", "-m", "Update cross-dataset comparison chart"],
                cwd=tmp_dir, check=True, capture_output=True, text=True,
            )
            subprocess.run(["git", "push"], cwd=tmp_dir, check=True,
                            capture_output=True, text=True)
            print("  Pushed to wiki successfully")
        else:
            print("  No changes to push (image unchanged)")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-dataset comparison tool")
    parser.add_argument("--output", type=str, default="cross-dataset-comparison.png",
                        help="Output chart path (default: cross-dataset-comparison.png)")
    parser.add_argument("--summary", action="store_true",
                        help="Print text summary only (no chart)")
    parser.add_argument("--wiki", action="store_true",
                        help="Generate chart and push to wiki repo")

    args = parser.parse_args()

    runs = load_all_results()

    if not runs:
        print("No results found in results/*/results.tsv")
        print("Run experiments first, or copy results.tsv files to results/<dataset>/")
        sys.exit(1)

    print_summary(runs)

    if not args.summary:
        generate_chart(runs, args.output)

        if args.wiki:
            push_to_wiki(args.output, runs=runs)


if __name__ == "__main__":
    main()
