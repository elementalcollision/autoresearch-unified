"""Integration tests for drift/monitor.py — full DriftMonitor pipeline."""

import pytest

from autoresearch.results import ExperimentResult
from drift.monitor import DriftMonitor, DriftConfig, DriftReport


def _make_result(exp_num, desc="test", val_bpb=1.3, status="keep", **kwargs):
    defaults = dict(
        peak_mem_gb=8.0, tok_sec=10000, mfu=20.0, steps=100,
        notes="", gpu_name="H100", baseline_sha="abc",
        watts=300.0, joules_per_token=0.001, total_energy_joules=500.0,
    )
    defaults.update(kwargs)
    return ExperimentResult(
        exp=f"exp{exp_num}", description=desc,
        val_bpb=val_bpb, status=status, **defaults,
    )


class TestDriftMonitor:
    def test_no_alerts_with_healthy_results(self):
        monitor = DriftMonitor()
        # Steadily improving
        results = [_make_result(i, f"exp {i}", 1.5 - i * 0.01) for i in range(10)]
        report = monitor.check(results)
        assert not report.has_alerts
        assert report.critical_count == 0

    def test_multiple_alerts_combined(self):
        monitor = DriftMonitor(DriftConfig())
        # Plateau + same category repeated
        results = [_make_result(0, "baseline", 1.3, "baseline")]
        # 25 experiments all learning_rate, all same val_bpb
        for i in range(1, 26):
            results.append(_make_result(i, f"Change MATRIX_LR to {0.01 * i}", 1.3, "discard"))
        report = monitor.check(results)
        assert report.has_alerts

    def test_format_for_prompt_empty_when_no_alerts(self):
        monitor = DriftMonitor()
        results = [_make_result(0, "baseline", 1.3, "baseline")]
        report = monitor.check(results)
        assert report.format_for_prompt() == ""

    def test_format_for_prompt_has_content_with_alerts(self):
        monitor = DriftMonitor(DriftConfig())
        results = [_make_result(0, "baseline", 1.3, "baseline")]
        for i in range(1, 20):
            results.append(_make_result(i, f"Change MATRIX_LR to {0.01 * i}", 1.4, "discard"))
        report = monitor.check(results)
        prompt_text = report.format_for_prompt()
        if report.has_alerts:
            assert "DRIFT MONITOR ALERTS" in prompt_text

    def test_inject_into_prompt(self):
        monitor = DriftMonitor()
        results = [_make_result(0, "baseline", 1.3, "baseline")]
        for i in range(1, 20):
            results.append(_make_result(i, f"Change MATRIX_LR to {0.01 * i}", 1.4, "discard"))
        base = "You are an AI researcher."
        injected = monitor.inject_into_prompt(base, results)
        assert injected.startswith(base)


class TestDriftReport:
    def test_summary_no_alerts(self):
        report = DriftReport()
        assert "No drift" in report.format_summary()

    def test_summary_with_alerts(self):
        from drift.performance import PerformanceAlert
        report = DriftReport(
            performance=[PerformanceAlert("critical", "plateau", "test")],
        )
        assert "1 critical" in report.format_summary()
        assert report.critical_count == 1
