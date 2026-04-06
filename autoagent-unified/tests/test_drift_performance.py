"""Tests for drift/performance.py — plateau and regression detection."""

import pytest

from autoresearch.results import ExperimentResult
from drift.performance import PerformanceDrift, PerformanceConfig


def _make_result(exp_num, val_bpb, status="keep", tok_sec=10000, **kwargs):
    return ExperimentResult(
        exp=f"exp{exp_num}", description=f"test {exp_num}",
        val_bpb=val_bpb, peak_mem_gb=8.0, tok_sec=tok_sec,
        mfu=20.0, steps=100, status=status, notes="",
        **kwargs,
    )


class TestPlateauDetection:
    def test_no_plateau_with_few_results(self):
        drift = PerformanceDrift(PerformanceConfig(plateau_window=20))
        results = [_make_result(i, 1.5 - i * 0.01) for i in range(5)]
        alerts = drift.check(results)
        assert not any(a.category == "plateau" for a in alerts)

    def test_plateau_detected_with_flat_results(self):
        drift = PerformanceDrift(PerformanceConfig(plateau_window=10, plateau_slope_threshold=0.001))
        # 20 experiments all at nearly the same val_bpb
        results = [_make_result(i, 1.300 + (i % 3) * 0.0001) for i in range(20)]
        alerts = drift.check(results)
        plateau_alerts = [a for a in alerts if a.category == "plateau"]
        assert len(plateau_alerts) > 0

    def test_no_plateau_with_improving_results(self):
        drift = PerformanceDrift(PerformanceConfig(plateau_window=10))
        # Steady improvement
        results = [_make_result(i, 1.5 - i * 0.01) for i in range(20)]
        alerts = drift.check(results)
        plateau_alerts = [a for a in alerts if a.category == "plateau"]
        assert len(plateau_alerts) == 0


class TestRegressionDetection:
    def test_regression_detected(self):
        drift = PerformanceDrift(PerformanceConfig(regression_patience=5))
        results = [_make_result(0, 1.3, "baseline")]
        # 10 experiments that don't beat baseline
        results += [_make_result(i, 1.4, "discard") for i in range(1, 11)]
        alerts = drift.check(results)
        regression = [a for a in alerts if a.category == "regression"]
        assert len(regression) > 0

    def test_no_regression_with_recent_improvement(self):
        drift = PerformanceDrift(PerformanceConfig(regression_patience=5))
        results = [_make_result(0, 1.3, "baseline")]
        results += [_make_result(i, 1.4, "discard") for i in range(1, 4)]
        results.append(_make_result(4, 1.25, "keep"))  # improvement!
        alerts = drift.check(results)
        regression = [a for a in alerts if a.category == "regression"]
        assert len(regression) == 0


class TestThroughputAnomaly:
    def test_throughput_drop_detected(self):
        drift = PerformanceDrift(PerformanceConfig(throughput_drop_pct=10.0))
        results = [_make_result(i, 1.3, tok_sec=10000) for i in range(10)]
        results.append(_make_result(10, 1.3, tok_sec=5000))  # 50% drop
        alerts = drift.check(results)
        throughput = [a for a in alerts if a.category == "throughput"]
        assert len(throughput) > 0


class TestKeepRatio:
    def test_low_keep_ratio_warning(self):
        drift = PerformanceDrift(PerformanceConfig(keep_ratio_warning=0.15))
        results = [_make_result(0, 1.3, "baseline")]
        # 19 discards, 1 keep
        results += [_make_result(i, 1.4, "discard") for i in range(1, 20)]
        results.append(_make_result(20, 1.25, "keep"))
        alerts = drift.check(results)
        keep_alerts = [a for a in alerts if "keep ratio" in a.message.lower()]
        assert len(keep_alerts) > 0
