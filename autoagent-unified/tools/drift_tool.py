"""Drift monitor tool for autoagent's agent harness.

Provides drift detection status to the agent so it can self-correct.
"""

import os

from autoresearch.results import load_results
from drift.monitor import DriftMonitor, DriftConfig


RESULTS_PATH = os.environ.get("RESULTS_TSV", "results.tsv")


def check_drift_status(detail_level: str = "summary") -> str:
    """Get current drift-monitor report.

    Args:
        detail_level: "summary" for a one-line status, "full" for all alerts.

    Returns:
        Drift report as formatted text.
    """
    results = load_results(RESULTS_PATH)
    if not results:
        return "No experiments yet — drift monitoring not active."

    monitor = DriftMonitor(DriftConfig())
    report = monitor.check(results)

    if detail_level == "full":
        if not report.has_alerts:
            return "No drift detected. All systems nominal."

        lines = [f"Drift Report ({report.critical_count} critical, {report.warning_count} warnings):"]
        lines.append("")

        for alert in report.all_alerts:
            icon = {"critical": "[!]", "warning": "[~]", "info": "[i]"}.get(alert.level, "[?]")
            lines.append(f"  {icon} [{alert.category}] {alert.message}")

        return "\n".join(lines)

    # Summary
    return report.format_summary()
