#!/usr/bin/env python3
"""Statistical analysis: normalization curves and reproducibility evidence.

Analyzes completed experiment runs to produce:
1. Normalized convergence curves (% improvement from baseline)
2. Phase transition detection (LR plateau → architectural breakthrough)
3. LR-ceiling statistical tests
4. Cross-platform normalization
5. Hyperparameter convergence analysis
6. Reproducibility metrics with CV%

Usage:
    python scripts/normalization_analysis.py [--output-dir OUTPUT_DIR]
"""

import argparse
import csv
import os
import sys
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

def load_tsv(path: str) -> list[Result]:
    results = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                results.append(Result(
                    exp=row["exp"],
                    description=row.get("description", ""),
                    val_bpb=float(row.get("val_bpb", 0)),
                    peak_mem_gb=float(row.get("peak_mem_gb", 0)),
                    tok_sec=int(float(row.get("tok_sec", 0))),
                    mfu=float(row.get("mfu", 0)),
                    steps=int(float(row.get("steps", 0))),
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                    gpu_name=row.get("gpu_name", ""),
                ))
            except (ValueError, KeyError):
                continue
    return results


def exp_num(r: Result) -> int:
    return int(r.exp.replace("exp", ""))


# ---------------------------------------------------------------------------
# Analysis 1: Normalized convergence curves
# ---------------------------------------------------------------------------

def compute_convergence_curve(results: list[Result]) -> list[tuple[int, float, float]]:
    """Compute cummin val_bpb normalized as % improvement from baseline.

    Returns list of (exp_number, val_bpb, pct_improvement).
    """
    baseline = results[0].val_bpb
    if baseline <= 0:
        return []
    best_so_far = baseline
    curve = []
    for r in results:
        if r.val_bpb > 0 and r.val_bpb < best_so_far:
            best_so_far = r.val_bpb
        pct = (baseline - best_so_far) / baseline * 100
        curve.append((exp_num(r), best_so_far, pct))
    return curve


# ---------------------------------------------------------------------------
# Analysis 2: Phase transition detection
# ---------------------------------------------------------------------------

def detect_phases(results: list[Result], gap_threshold: int = 10) -> dict:
    """Detect optimization phases from the experiment trajectory.

    Phases:
        1. LR tuning — initial rapid improvement
        2. Plateau — gap with no keeps
        3. Breakthrough — architectural change (e.g., WINDOW_PATTERN)
        4. Second tuning — re-optimization on new architecture
        5. Final plateau — convergence
    """
    keeps = [(exp_num(r), r) for r in results if r.status in ("keep", "baseline")]
    if len(keeps) < 2:
        return {"phases": [], "plateau_start": None, "breakthrough": None}

    # Find first gap of >= gap_threshold experiments between keeps
    plateau_start = None
    breakthrough = None
    for i in range(1, len(keeps)):
        gap = keeps[i][0] - keeps[i - 1][0]
        if gap >= gap_threshold and plateau_start is None:
            plateau_start = keeps[i - 1][0]
            # The next keep after the gap is the breakthrough
            breakthrough_exp = keeps[i][0]
            breakthrough_r = keeps[i][1]
            # Check if improvement is > 0.3%
            pre_best = min(k[1].val_bpb for k in keeps[:i] if k[1].val_bpb > 0)
            post_bpb = breakthrough_r.val_bpb
            if post_bpb > 0 and pre_best > 0:
                delta_pct = (pre_best - post_bpb) / pre_best * 100
                breakthrough = {
                    "exp": breakthrough_exp,
                    "description": breakthrough_r.description,
                    "val_bpb": post_bpb,
                    "pre_best": pre_best,
                    "delta_pct": delta_pct,
                }
            break

    # Classify phases
    baseline_bpb = results[0].val_bpb
    phases = []

    if plateau_start is not None:
        # Phase 1: exp0 to plateau_start
        p1_keeps = [k for k in keeps if k[0] <= plateau_start]
        p1_best = min(k[1].val_bpb for k in p1_keeps if k[1].val_bpb > 0)
        phases.append({
            "name": "LR tuning",
            "start": 0,
            "end": plateau_start,
            "keeps": len(p1_keeps) - 1,  # exclude baseline
            "best_bpb": p1_best,
            "improvement_pct": (baseline_bpb - p1_best) / baseline_bpb * 100,
        })

    if breakthrough:
        # Plateau
        phases.append({
            "name": "Plateau",
            "start": plateau_start + 1 if plateau_start else 0,
            "end": breakthrough["exp"] - 1,
            "keeps": 0,
            "best_bpb": None,
            "improvement_pct": 0,
        })

        # Phase 2: breakthrough onward
        p2_keeps = [k for k in keeps if k[0] >= breakthrough["exp"]]
        p2_end = keeps[-1][0] if keeps else 99
        total_exp = exp_num(results[-1])
        p2_best = min(k[1].val_bpb for k in p2_keeps if k[1].val_bpb > 0) if p2_keeps else None
        phases.append({
            "name": "Post-breakthrough",
            "start": breakthrough["exp"],
            "end": p2_end,
            "keeps": len(p2_keeps),
            "best_bpb": p2_best,
            "improvement_pct": (baseline_bpb - p2_best) / baseline_bpb * 100 if p2_best else 0,
        })

        # Final plateau (if any)
        if total_exp > p2_end + 3:
            phases.append({
                "name": "Final plateau",
                "start": p2_end + 1,
                "end": total_exp,
                "keeps": 0,
                "best_bpb": None,
                "improvement_pct": 0,
            })

    return {
        "phases": phases,
        "plateau_start": plateau_start,
        "breakthrough": breakthrough,
    }


# ---------------------------------------------------------------------------
# Analysis 3: LR-ceiling statistical test
# ---------------------------------------------------------------------------

def lr_ceiling_test(ceiling_values: list[float]) -> dict:
    """Test whether pre-breakthrough val_bpb values cluster around a ceiling."""
    arr = np.array(ceiling_values)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = (std / mean * 100) if mean > 0 else 0.0

    result = {
        "n": len(arr),
        "mean": mean,
        "std": std,
        "cv_pct": cv,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
    }

    if len(arr) >= 3:
        from scipy import stats as sp_stats
        t_stat, p_value = sp_stats.ttest_1samp(arr, 1.078)
        sem = sp_stats.sem(arr)
        ci = sp_stats.t.interval(0.95, len(arr) - 1, loc=mean, scale=sem)
        result.update({
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "ci_95_lower": float(ci[0]),
            "ci_95_upper": float(ci[1]),
        })

    return result


# ---------------------------------------------------------------------------
# Analysis 4: Cross-platform normalization
# ---------------------------------------------------------------------------

def normalize_run(results: list[Result], gpu_vram_gb: float) -> dict:
    """Compute normalized metrics for a single run."""
    baseline_bpb = results[0].val_bpb
    keeps = [r for r in results if r.status in ("keep", "baseline") and r.val_bpb > 0]
    best_bpb = min(r.val_bpb for r in keeps) if keeps else baseline_bpb
    crashes = sum(1 for r in results if r.status == "crash")
    total = len(results)

    # Average throughput from kept experiments
    kept_toksec = [r.tok_sec for r in keeps if r.tok_sec > 0]
    avg_tok_sec = np.mean(kept_toksec) if kept_toksec else 0

    improvement_pct = (baseline_bpb - best_bpb) / baseline_bpb * 100

    return {
        "baseline_bpb": baseline_bpb,
        "best_bpb": best_bpb,
        "improvement_pct": improvement_pct,
        "total_experiments": total,
        "kept": len(keeps),
        "keep_rate": len(keeps) / total * 100,
        "crashes": crashes,
        "crash_rate": crashes / total * 100,
        "efficiency": improvement_pct / total,  # improvement per experiment
        "avg_tok_sec": float(avg_tok_sec),
        "vram_normalized_throughput": float(avg_tok_sec / gpu_vram_gb),
    }


# ---------------------------------------------------------------------------
# Analysis 5: Hyperparameter convergence
# ---------------------------------------------------------------------------

HP_KEYWORDS = {
    "WEIGHT_DECAY": ["weight_decay"],
    "WARMDOWN_RATIO": ["warmdown_ratio", "warmdown"],
    "MATRIX_LR": ["matrix_lr"],
    "SCALAR_LR": ["scalar_lr"],
    "EMBEDDING_LR": ["embedding_lr"],
    "UNEMBEDDING_LR": ["unembedding_lr"],
    "MLP_RATIO": ["mlp_ratio", "mlp ratio"],
    "ASPECT_RATIO": ["aspect_ratio"],
    "WINDOW_PATTERN": ["window_pattern"],
    "HEAD_DIM": ["head_dim"],
    "DEPTH": ["depth"],
    "FINAL_LR_FRAC": ["final_lr_frac"],
    "BATCH_SIZE": ["batch_size", "total_batch", "device_batch"],
}


def classify_hp(description: str) -> str:
    desc_lower = description.lower()
    for hp, keywords in HP_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return hp
    return "OTHER"


def hp_contribution_analysis(results: list[Result]) -> list[dict]:
    """Compute per-hyperparameter improvement contribution."""
    contributions = []
    baseline_bpb = results[0].val_bpb
    keeps = [r for r in results if r.status == "keep" and r.val_bpb > 0]

    prev_bpb = baseline_bpb
    for r in keeps:
        delta_pct = (prev_bpb - r.val_bpb) / baseline_bpb * 100
        contributions.append({
            "exp": r.exp,
            "hp": classify_hp(r.description),
            "description": r.description,
            "val_bpb": r.val_bpb,
            "delta_pct": delta_pct,
            "cumulative_pct": (baseline_bpb - r.val_bpb) / baseline_bpb * 100,
        })
        prev_bpb = r.val_bpb

    return contributions


# ---------------------------------------------------------------------------
# Analysis 6: Reproducibility metrics
# ---------------------------------------------------------------------------

def reproducibility_table(run_metrics: list[dict]) -> dict:
    """Compute mean, SD, and CV% across runs for key metrics."""
    metrics = {}
    keys = ["baseline_bpb", "best_bpb", "improvement_pct", "kept", "keep_rate",
            "crashes", "crash_rate", "efficiency"]

    for key in keys:
        values = [m[key] for m in run_metrics if key in m]
        arr = np.array(values, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        cv = (std / mean * 100) if mean != 0 else 0.0
        metrics[key] = {"mean": mean, "std": std, "cv_pct": cv, "values": values}

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_data_files(base_dir: Path) -> dict[str, Path]:
    """Discover available result TSV files."""
    files = {}
    results_dir = base_dir / "results" / "pubmed"

    # 5090 runs
    for i in range(1, 4):
        p = results_dir / f"results_gpt41_5090_r{i}.tsv"
        if p.exists():
            files[f"5090_R{i}"] = p

    # PR#17 5070Ti runs
    pr17_dir = results_dir / "bmdhodl-rtx5070ti"
    for i in range(1, 4):
        p = pr17_dir / f"results_gpt41_r{i}.tsv"
        if p.exists():
            files[f"5070Ti_R{i}"] = p

    return files


def main():
    parser = argparse.ArgumentParser(description="Normalization analysis")
    parser.add_argument("--output-dir", default="docs", help="Output directory")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / args.output_dir
    output_dir.mkdir(exist_ok=True)

    files = find_data_files(base_dir)
    if not files:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} result files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    print()

    # Load all runs
    runs = {}
    for name, path in files.items():
        runs[name] = load_tsv(str(path))
        print(f"  {name}: {len(runs[name])} experiments")

    # -----------------------------------------------------------------------
    # Analysis 1: Convergence curves
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 1: NORMALIZED CONVERGENCE CURVES")
    print("=" * 80)

    for name, results in runs.items():
        curve = compute_convergence_curve(results)
        if not curve:
            continue
        print(f"\n  {name}:")
        print(f"  {'Exp':>5} {'val_bpb':>10} {'Improvement':>12}")
        print(f"  {'-'*5} {'-'*10} {'-'*12}")
        # Print every 10th point + final
        for i, (exp_n, bpb, pct) in enumerate(curve):
            if exp_n % 10 == 0 or i == len(curve) - 1:
                print(f"  {exp_n:>5} {bpb:>10.6f} {pct:>+11.3f}%")

    # -----------------------------------------------------------------------
    # Analysis 2: Phase detection
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 2: PHASE TRANSITION DETECTION")
    print("=" * 80)

    phase_data = {}
    for name, results in runs.items():
        pd = detect_phases(results)
        phase_data[name] = pd
        print(f"\n  {name}:")
        if pd["breakthrough"]:
            b = pd["breakthrough"]
            print(f"    Breakthrough at exp{b['exp']}: {b['description']}")
            print(f"    Impact: {b['pre_best']:.6f} → {b['val_bpb']:.6f} ({b['delta_pct']:+.3f}%)")
        else:
            print(f"    No clear breakthrough detected")
        for phase in pd["phases"]:
            span = phase["end"] - phase["start"] + 1
            print(f"    Phase '{phase['name']}': exp{phase['start']}-{phase['end']} "
                  f"({span} exp, {phase['keeps']} keeps, {phase['improvement_pct']:+.3f}%)")

    # -----------------------------------------------------------------------
    # Analysis 3: LR-ceiling test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 3: LR-CEILING CONVERGENCE TEST")
    print("=" * 80)

    ceiling_values = []
    for name, pd in phase_data.items():
        if "5090" in name and pd["phases"]:
            p1 = pd["phases"][0]
            if p1["best_bpb"]:
                ceiling_values.append(p1["best_bpb"])
                print(f"  {name} LR-phase best: {p1['best_bpb']:.6f}")

    if ceiling_values:
        test = lr_ceiling_test(ceiling_values)
        print(f"\n  LR-Ceiling Statistics (n={test['n']}):")
        print(f"    Mean: {test['mean']:.6f}")
        print(f"    SD:   {test['std']:.6f}")
        print(f"    CV:   {test['cv_pct']:.4f}%")
        print(f"    Range: {test['range']:.6f}")
        if "p_value" in test:
            print(f"    t-test vs 1.078: t={test['t_stat']:.4f}, p={test['p_value']:.6f}")
            print(f"    95% CI: [{test['ci_95_lower']:.6f}, {test['ci_95_upper']:.6f}]")

    # -----------------------------------------------------------------------
    # Analysis 4: Cross-platform normalization
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 4: CROSS-PLATFORM NORMALIZATION")
    print("=" * 80)

    vram = {"5090": 32.0, "5070Ti": 16.0}
    normalized = {}
    for name, results in runs.items():
        gpu = "5090" if "5090" in name else "5070Ti"
        nm = normalize_run(results, vram[gpu])
        normalized[name] = nm

    print(f"\n  {'Run':<12} {'Baseline':>10} {'Best':>10} {'Δ%':>8} {'Keep%':>7} "
          f"{'Crash%':>8} {'Eff':>8} {'tok/s/GB':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*8} {'-'*10}")
    for name, nm in normalized.items():
        print(f"  {name:<12} {nm['baseline_bpb']:>10.6f} {nm['best_bpb']:>10.6f} "
              f"{nm['improvement_pct']:>+7.2f}% {nm['keep_rate']:>6.1f}% "
              f"{nm['crash_rate']:>7.1f}% {nm['efficiency']:>7.4f} "
              f"{nm['vram_normalized_throughput']:>9.0f}")

    # -----------------------------------------------------------------------
    # Analysis 5: Hyperparameter contributions
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 5: HYPERPARAMETER CONTRIBUTION ANALYSIS")
    print("=" * 80)

    for name, results in runs.items():
        if "5090" not in name:
            continue
        contribs = hp_contribution_analysis(results)
        print(f"\n  {name}:")
        print(f"  {'Exp':<6} {'HP':<16} {'Δ%':>8} {'Cum%':>8}  Description")
        print(f"  {'-'*6} {'-'*16} {'-'*8} {'-'*8}  {'-'*40}")
        for c in contribs:
            print(f"  {c['exp']:<6} {c['hp']:<16} {c['delta_pct']:>+7.3f}% "
                  f"{c['cumulative_pct']:>+7.3f}%  {c['description'][:50]}")

        # Category summary
        categories = {}
        for c in contribs:
            hp = c["hp"]
            if hp not in categories:
                categories[hp] = {"count": 0, "total_delta": 0.0}
            categories[hp]["count"] += 1
            categories[hp]["total_delta"] += c["delta_pct"]

        print(f"\n  Category Summary:")
        print(f"  {'Category':<16} {'Keeps':>6} {'Total Δ%':>10}")
        for hp, data in sorted(categories.items(), key=lambda x: -x[1]["total_delta"]):
            print(f"  {hp:<16} {data['count']:>6} {data['total_delta']:>+9.3f}%")

    # -----------------------------------------------------------------------
    # Analysis 6: Reproducibility metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ANALYSIS 6: REPRODUCIBILITY METRICS")
    print("=" * 80)

    # 5090 runs
    r5090 = {k: v for k, v in normalized.items() if "5090" in k}
    if len(r5090) >= 2:
        repro = reproducibility_table(list(r5090.values()))
        print(f"\n  5090 Runs (n={len(r5090)}):")
        print(f"  {'Metric':<20} {'Mean':>10} {'SD':>10} {'CV%':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for key, data in repro.items():
            fmt = ".6f" if "bpb" in key else ".2f"
            print(f"  {key:<20} {data['mean']:>10{fmt}} {data['std']:>10{fmt}} "
                  f"{data['cv_pct']:>7.2f}%")

    # 5070Ti runs
    r5070 = {k: v for k, v in normalized.items() if "5070" in k}
    if len(r5070) >= 2:
        repro = reproducibility_table(list(r5070.values()))
        print(f"\n  5070Ti Runs (n={len(r5070)}):")
        print(f"  {'Metric':<20} {'Mean':>10} {'SD':>10} {'CV%':>8}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*8}")
        for key, data in repro.items():
            fmt = ".6f" if "bpb" in key else ".2f"
            print(f"  {key:<20} {data['mean']:>10{fmt}} {data['std']:>10{fmt}} "
                  f"{data['cv_pct']:>7.2f}%")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY: KEY REPRODUCIBILITY EVIDENCE")
    print("=" * 80)

    if len(ceiling_values) >= 2:
        print(f"\n  1. LR-only ceiling: {np.mean(ceiling_values):.6f} ± {np.std(ceiling_values, ddof=1):.6f} "
              f"(CV = {np.std(ceiling_values, ddof=1)/np.mean(ceiling_values)*100:.4f}%)")

    window_exps = []
    for name, pd in phase_data.items():
        if "5090" in name and pd["breakthrough"]:
            window_exps.append(pd["breakthrough"]["exp"])
    if window_exps:
        print(f"  2. WINDOW_PATTERN discovered independently at: {window_exps} "
              f"(mean = exp{np.mean(window_exps):.0f})")

    print(f"  3. Total experiments analyzed: {sum(len(r) for r in runs.values())}")
    print(f"  4. Platforms: {len(set('5090' if '5090' in n else '5070Ti' for n in runs))}")
    print(f"  5. Runs: {len(runs)}")


if __name__ == "__main__":
    main()
