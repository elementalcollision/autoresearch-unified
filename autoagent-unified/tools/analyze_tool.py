"""Result analysis tool for autoagent's agent harness.

Wraps autoresearch's results module to provide experiment history analysis.
"""

import json
import os

from autoresearch.results import (
    categorize_experiments,
    format_history_for_prompt,
    get_best_result,
    load_results,
)


RESULTS_PATH = os.environ.get("RESULTS_TSV", "results.tsv")


def analyze_experiment_history(query: str = "summary") -> str:
    """Query results.tsv — trends, best results, category breakdown.

    Args:
        query: One of "summary", "best", "categories", "full", or "recent".

    Returns:
        Formatted analysis text.
    """
    results = load_results(RESULTS_PATH)

    if not results:
        return "No experiments recorded yet."

    query = query.lower().strip()

    if query == "best":
        best_bpb, best_exp = get_best_result(RESULTS_PATH)
        if best_bpb == float("inf"):
            return "No successful experiments yet."
        return f"Best val_bpb: {best_bpb:.6f} from {best_exp}"

    if query == "categories":
        categories = categorize_experiments(results)
        if not categories:
            return "No non-baseline experiments to categorize."
        lines = ["Strategy category breakdown:"]
        for cat, count in sorted(categories.items()):
            kept = sum(
                1 for r in results
                if r.status == "keep" and r.description.lower()
                and any(k in r.description.lower() for k in [cat])
            )
            lines.append(f"  {cat}: {count} tried")
        return "\n".join(lines)

    if query == "recent":
        recent = results[-10:]
        lines = [f"Last {len(recent)} experiments:"]
        for r in recent:
            bpb = f"{r.val_bpb:.4f}" if r.val_bpb > 0 else "crash"
            lines.append(f"  {r.exp} [{r.status}] val_bpb={bpb} — {r.description}")
        return "\n".join(lines)

    if query == "full":
        return format_history_for_prompt(RESULTS_PATH)

    # Default: summary
    total = len(results)
    kept = sum(1 for r in results if r.status == "keep")
    discarded = sum(1 for r in results if r.status == "discard")
    crashes = sum(1 for r in results if r.status == "crash")
    best_bpb, best_exp = get_best_result(RESULTS_PATH)

    summary = {
        "total_experiments": total,
        "kept": kept,
        "discarded": discarded,
        "crashes": crashes,
        "best_val_bpb": best_bpb if best_bpb < float("inf") else None,
        "best_experiment": best_exp,
        "keep_ratio": f"{kept / max(total - 1, 1):.1%}",
    }
    return json.dumps(summary, indent=2)
