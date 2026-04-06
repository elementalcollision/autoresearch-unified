"""Performance drift detection — val_bpb plateaus, throughput regressions, keep/discard trends."""

from dataclasses import dataclass, field

from autoresearch.results import ExperimentResult


@dataclass
class PerformanceAlert:
    """A single performance drift alert."""
    level: str       # "info", "warning", "critical"
    category: str    # "plateau", "regression", "throughput"
    message: str


@dataclass
class PerformanceConfig:
    """Tunable thresholds for performance drift detection."""
    plateau_window: int = 20          # look at last N kept experiments
    plateau_slope_threshold: float = 0.0005  # |slope| below this = plateau
    regression_patience: int = 15     # no improvement for N experiments = regression
    throughput_drop_pct: float = 10.0 # tok/sec drop > this % = anomaly
    keep_ratio_warning: float = 0.15  # keep ratio below this = struggling


class PerformanceDrift:
    """Detects performance-related drift in experiment results."""

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()

    def check(self, results: list[ExperimentResult]) -> list[PerformanceAlert]:
        alerts = []
        alerts.extend(self._check_plateau(results))
        alerts.extend(self._check_regression(results))
        alerts.extend(self._check_throughput(results))
        alerts.extend(self._check_keep_ratio(results))
        return alerts

    def _check_plateau(self, results: list[ExperimentResult]) -> list[PerformanceAlert]:
        """Linear regression on recent kept experiments to detect plateau."""
        kept = [r for r in results if r.status == "keep" and r.val_bpb > 0]
        if len(kept) < self.config.plateau_window:
            return []

        recent = kept[-self.config.plateau_window:]
        n = len(recent)
        xs = list(range(n))
        ys = [r.val_bpb for r in recent]

        # Simple linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        sum_x = sum(xs)
        sum_y = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if denom == 0:
            return []

        slope = (n * sum_xy - sum_x * sum_y) / denom

        if abs(slope) < self.config.plateau_slope_threshold:
            return [PerformanceAlert(
                level="warning",
                category="plateau",
                message=f"Performance plateau detected: val_bpb slope is {slope:.6f} "
                        f"over last {n} kept experiments (threshold: {self.config.plateau_slope_threshold}). "
                        f"Consider trying a fundamentally different strategy category.",
            )]
        return []

    def _check_regression(self, results: list[ExperimentResult]) -> list[PerformanceAlert]:
        """Check if best val_bpb hasn't improved in N experiments."""
        if not results:
            return []

        best_bpb = float("inf")
        best_idx = -1
        for i, r in enumerate(results):
            if r.status in ("keep", "baseline") and r.val_bpb > 0 and r.val_bpb < best_bpb:
                best_bpb = r.val_bpb
                best_idx = i

        if best_idx < 0:
            return []

        experiments_since_best = len(results) - best_idx - 1
        if experiments_since_best >= self.config.regression_patience:
            return [PerformanceAlert(
                level="critical",
                category="regression",
                message=f"No improvement in {experiments_since_best} experiments since best "
                        f"val_bpb={best_bpb:.4f}. The optimization may be stuck in a local optimum. "
                        f"Consider a large architectural change or different hyperparameter axis.",
            )]
        return []

    def _check_throughput(self, results: list[ExperimentResult]) -> list[PerformanceAlert]:
        """Check for sudden tok/sec drops."""
        recent = [r for r in results if r.tok_sec > 0]
        if len(recent) < 5:
            return []

        # Rolling average of last 10 vs last experiment
        window = recent[-10:]
        avg_toksec = sum(r.tok_sec for r in window) / len(window)
        latest = recent[-1].tok_sec

        if avg_toksec > 0:
            drop_pct = (avg_toksec - latest) / avg_toksec * 100
            if drop_pct > self.config.throughput_drop_pct:
                return [PerformanceAlert(
                    level="warning",
                    category="throughput",
                    message=f"Throughput drop: latest {latest} tok/sec vs rolling avg "
                            f"{avg_toksec:.0f} tok/sec ({drop_pct:.1f}% drop). "
                            f"Possible hardware issue or batch size change.",
                )]
        return []

    def _check_keep_ratio(self, results: list[ExperimentResult]) -> list[PerformanceAlert]:
        """Check if recent experiments have a very low keep ratio."""
        recent = [r for r in results[-20:] if r.status in ("keep", "discard", "crash")]
        if len(recent) < 10:
            return []

        kept = sum(1 for r in recent if r.status == "keep")
        ratio = kept / len(recent)

        if ratio < self.config.keep_ratio_warning:
            return [PerformanceAlert(
                level="warning",
                category="plateau",
                message=f"Low keep ratio: {kept}/{len(recent)} ({ratio:.0%}) recent experiments kept. "
                        f"The optimization strategy may need a fundamental shift.",
            )]
        return []
