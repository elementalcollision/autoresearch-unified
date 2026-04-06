# Ported from autoresearch-unified/tui/results.py (MIT)
"""Results management for the experiment orchestrator.

Handles reading, writing, and formatting results.tsv for the experiment loop.
Uses atomic writes for crash safety — see autoresearch/resilience.py.
"""

import csv
import os
from dataclasses import dataclass

from autoresearch.resilience import atomic_append, atomic_write, validate_results_tsv


@dataclass
class ExperimentResult:
    """A single experiment result."""
    exp: str           # "exp0", "exp1", etc.
    description: str   # short description
    val_bpb: float     # validation bits per byte (0.0 for crashes)
    peak_mem_gb: float # peak memory in GB (0.0 for crashes)
    tok_sec: int       # tokens per second
    mfu: float         # model FLOPS utilization %
    steps: int         # training steps completed
    status: str        # "baseline", "keep", "discard", "crash", "skip"
    notes: str         # extra notes
    gpu_name: str = "" # hardware fingerprint
    baseline_sha: str = ""  # commit SHA of the unmodified training script
    watts: float = 0.0              # average power draw during training (W)
    joules_per_token: float = 0.0   # energy per token (J/tok)
    total_energy_joules: float = 0.0  # total energy for the experiment (J)


HEADER = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\n"


def init_results_tsv(path: str = "results.tsv") -> None:
    """Create results.tsv with header if it doesn't exist."""
    if not os.path.exists(path):
        atomic_write(path, HEADER)
    else:
        is_valid, warnings = validate_results_tsv(path)
        if warnings:
            import sys
            for w in warnings:
                print(f"  results.tsv: {w}", file=sys.stderr, flush=True)


def append_result(path: str, result: ExperimentResult) -> None:
    """Append an experiment result to results.tsv (crash-safe)."""
    if not os.path.exists(path):
        init_results_tsv(path)

    line = (
        f"{result.exp}\t"
        f"{result.description}\t"
        f"{result.val_bpb:.6f}\t"
        f"{result.peak_mem_gb:.1f}\t"
        f"{result.tok_sec}\t"
        f"{result.mfu:.1f}\t"
        f"{result.steps}\t"
        f"{result.status}\t"
        f"{result.notes}\t"
        f"{result.gpu_name}\t"
        f"{result.baseline_sha}\t"
        f"{result.watts:.1f}\t"
        f"{result.joules_per_token:.6f}\t"
        f"{result.total_energy_joules:.1f}\n"
    )
    atomic_append(path, line)


def load_results(path: str = "results.tsv") -> list[ExperimentResult]:
    """Load all experiment results from results.tsv."""
    if not os.path.exists(path):
        return []

    results = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                results.append(ExperimentResult(
                    exp=row.get("exp", ""),
                    description=row.get("description", ""),
                    val_bpb=float(row.get("val_bpb", 0)),
                    peak_mem_gb=float(row.get("peak_mem_gb", 0)),
                    tok_sec=int(row.get("tok_sec", 0)),
                    mfu=float(row.get("mfu", 0)),
                    steps=int(row.get("steps", 0)),
                    status=row.get("status", ""),
                    notes=row.get("notes", ""),
                    gpu_name=row.get("gpu_name", ""),
                    baseline_sha=row.get("baseline_sha", ""),
                    watts=float(row.get("watts", 0) or 0),
                    joules_per_token=float(row.get("joules_per_token", 0) or 0),
                    total_energy_joules=float(row.get("total_energy_joules", 0) or 0),
                ))
            except (ValueError, TypeError):
                continue
    return results


def get_best_result(path: str = "results.tsv") -> tuple[float, str]:
    """Get the best val_bpb and experiment name from results."""
    results = load_results(path)
    best_bpb = float("inf")
    best_exp = "none"

    for r in results:
        if r.status in ("keep", "baseline") and r.val_bpb > 0:
            if r.val_bpb < best_bpb:
                best_bpb = r.val_bpb
                best_exp = f"{r.exp} ({r.description})"

    return best_bpb, best_exp


def next_experiment_number(path: str = "results.tsv") -> int:
    """Get the next experiment number based on existing results."""
    results = load_results(path)
    if not results:
        return 0

    max_num = -1
    for r in results:
        try:
            num = int(r.exp.replace("exp", ""))
            max_num = max(max_num, num)
        except ValueError:
            continue
    return max_num + 1


def classify_experiment(description: str) -> str:
    """Classify an experiment description into a strategy category."""
    desc = description.lower()

    if any(k in desc for k in ["batch_size", "batch size", "total_batch"]):
        return "batch_size"
    if any(k in desc for k in ["depth", "head_dim", "window_pattern",
                                "window pattern", "mlp_ratio", "aspect_ratio"]):
        return "architecture"
    if any(k in desc for k in ["warmup", "warmdown", "final_lr_frac",
                                "schedule", "cooldown"]):
        return "schedule"
    if any(k in desc for k in ["weight_decay", "weight decay", "adam_beta",
                                "regularization"]):
        return "regularization"
    if any(k in desc for k in ["_lr", "learning rate", "learning_rate",
                                "matrix_lr", "scalar_lr", "embedding_lr"]):
        return "learning_rate"
    if any(k in desc for k in ["activation_checkpointing", "compile_mode",
                                "compile mode"]):
        return "infrastructure"

    return "other"


def categorize_experiments(
    results: list[ExperimentResult],
) -> dict[str, int]:
    """Count experiments per strategy category."""
    counts: dict[str, int] = {}
    for r in results:
        if r.status == "baseline":
            continue
        cat = classify_experiment(r.description)
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def format_history_for_prompt(path: str = "results.tsv") -> str:
    """Format results as a readable table for the LLM prompt."""
    results = load_results(path)
    if not results:
        return "No experiments yet."

    lines = []
    lines.append(f"{'Exp':<6} {'Status':<9} {'val_bpb':>8} {'Mem(GB)':>8} {'Tok/s':>8} {'MFU':>6} {'Steps':>6}  Description")
    lines.append("-" * 90)

    for r in results:
        bpb_str = f"{r.val_bpb:.4f}" if r.val_bpb > 0 else "crash"
        lines.append(
            f"{r.exp:<6} {r.status:<9} {bpb_str:>8} {r.peak_mem_gb:>7.1f} "
            f"{r.tok_sec:>8} {r.mfu:>5.1f}% {r.steps:>6}  {r.description}"
        )

    # Strategy summary footer
    categories = categorize_experiments(results)
    if categories:
        lines.append("")
        lines.append("Strategy summary (category: tried / kept):")
        for cat, count in sorted(categories.items()):
            kept = sum(
                1 for r in results
                if r.status == "keep" and classify_experiment(r.description) == cat
            )
            lines.append(f"  {cat}: {count} tried, {kept} kept")

    return "\n".join(lines)
