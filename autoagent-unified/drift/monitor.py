"""Drift monitor orchestrator — combines performance, hardware, and strategy detectors."""

from dataclasses import dataclass, field

from autoresearch.results import ExperimentResult, load_results
from drift.performance import PerformanceDrift, PerformanceAlert, PerformanceConfig
from drift.hardware import HardwareDrift, HardwareAlert, HardwareConfig
from drift.strategy import StrategyDrift, StrategyAlert, StrategyConfig


@dataclass
class DriftConfig:
    """Combined configuration for all drift detectors."""
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


@dataclass
class DriftReport:
    """Combined drift report from all detectors."""
    performance: list[PerformanceAlert] = field(default_factory=list)
    hardware: list[HardwareAlert] = field(default_factory=list)
    strategy: list[StrategyAlert] = field(default_factory=list)

    @property
    def has_alerts(self) -> bool:
        return bool(self.performance or self.hardware or self.strategy)

    @property
    def all_alerts(self) -> list:
        return self.performance + self.hardware + self.strategy

    @property
    def critical_count(self) -> int:
        return sum(1 for a in self.all_alerts if a.level == "critical")

    @property
    def warning_count(self) -> int:
        return sum(1 for a in self.all_alerts if a.level == "warning")

    def format_for_prompt(self) -> str:
        """Format drift alerts as text to inject into the LLM system prompt."""
        if not self.has_alerts:
            return ""

        lines = ["\n--- DRIFT MONITOR ALERTS ---"]

        for alert in self.all_alerts:
            icon = {"critical": "CRITICAL", "warning": "WARNING", "info": "INFO"}.get(alert.level, "INFO")
            lines.append(f"[{icon}] {alert.message}")

        lines.append("--- END DRIFT ALERTS ---\n")
        return "\n".join(lines)

    def format_summary(self) -> str:
        """Short summary for status display."""
        if not self.has_alerts:
            return "No drift detected"
        parts = []
        if self.critical_count:
            parts.append(f"{self.critical_count} critical")
        if self.warning_count:
            parts.append(f"{self.warning_count} warnings")
        info_count = sum(1 for a in self.all_alerts if a.level == "info")
        if info_count:
            parts.append(f"{info_count} info")
        return f"Drift alerts: {', '.join(parts)}"


class DriftMonitor:
    """Orchestrates all drift detectors and produces combined reports.

    Usage:
        monitor = DriftMonitor()
        report = monitor.check(results)
        if report.has_alerts:
            prompt += report.format_for_prompt()
    """

    def __init__(self, config: DriftConfig | None = None):
        cfg = config or DriftConfig()
        self.perf = PerformanceDrift(cfg.performance)
        self.hw = HardwareDrift(cfg.hardware)
        self.strategy = StrategyDrift(cfg.strategy)

    def check(self, results: list[ExperimentResult]) -> DriftReport:
        """Run all drift checks against the experiment results."""
        return DriftReport(
            performance=self.perf.check(results),
            hardware=self.hw.check(results),
            strategy=self.strategy.check(results),
        )

    def check_from_file(self, results_path: str = "results.tsv") -> DriftReport:
        """Convenience: load results from TSV and run checks."""
        results = load_results(results_path)
        return self.check(results)

    def inject_into_prompt(self, base_prompt: str, results: list[ExperimentResult]) -> str:
        """Run drift checks and append any alerts to the system prompt."""
        report = self.check(results)
        if report.has_alerts:
            return base_prompt + report.format_for_prompt()
        return base_prompt
