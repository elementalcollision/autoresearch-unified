"""Results management for the experiment orchestrator.

Handles reading, writing, and formatting results.tsv for the experiment loop.
Uses atomic writes for crash safety — see tui/resilience.py.
"""

import csv
import os
from dataclasses import dataclass

from tui.resilience import atomic_append, atomic_write, validate_results_tsv


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
    status: str        # "baseline", "keep", "discard", "crash"
    notes: str         # extra notes
    gpu_name: str = "" # hardware fingerprint (e.g. "AMD Instinct MI300X OAM")
    baseline_sha: str = ""  # commit SHA of the unmodified training script
    watts: float = 0.0              # average power draw during training (W)
    joules_per_token: float = 0.0   # energy per token (J/tok)
    total_energy_joules: float = 0.0  # total energy for the experiment (J)


# TSV header — 14 columns. Columns 12-14 (watts, joules_per_token, total_energy_joules)
# track power/energy instrumentation for MLCommons/MLPerf Power analysis.
HEADER = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\n"


def init_results_tsv(path: str = "results.tsv") -> None:
    """Create results.tsv with header if it doesn't exist.

    If the file exists, validates it and fixes minor corruption
    (e.g., truncated trailing line from a mid-write crash).
    """
    if not os.path.exists(path):
        atomic_write(path, HEADER)
    else:
        # Validate and fix any corruption from previous crash
        is_valid, warnings = validate_results_tsv(path)
        if warnings:
            import sys
            for w in warnings:
                print(f"  \u26a0\ufe0f  results.tsv: {w}", file=sys.stderr, flush=True)


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
    """Load all experiment results from results.tsv.

    Skips corrupt lines gracefully (logs warning but doesn't crash).
    """
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
                    baseline_sha=row.get("baseline_sha", ""),  # backward-compat: empty for legacy TSVs
                    watts=float(row.get("watts", 0) or 0),
                    joules_per_token=float(row.get("joules_per_token", 0) or 0),
                    total_energy_joules=float(row.get("total_energy_joules", 0) or 0),
                ))
            except (ValueError, TypeError):
                continue
    return results


def get_best_result(path: str = "results.tsv") -> tuple[float, str]:
    """Get the best val_bpb and experiment name from results.

    Only considers "keep" and "baseline" experiments.
    Returns (best_val_bpb, best_exp_name). Returns (inf, "none") if no results.
    """
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

    return "\n".join(lines)
