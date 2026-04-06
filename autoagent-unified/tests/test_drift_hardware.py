"""Tests for drift/hardware.py — thermal throttling, memory, power drift."""

import pytest

from autoresearch.results import ExperimentResult
from drift.hardware import HardwareDrift, HardwareConfig


def _make_result(exp_num, peak_mem_gb=8.0, watts=300.0, tok_sec=10000, joules_per_token=0.001, val_bpb=1.3, **kwargs):
    return ExperimentResult(
        exp=f"exp{exp_num}", description=f"test {exp_num}",
        val_bpb=val_bpb, peak_mem_gb=peak_mem_gb, tok_sec=tok_sec,
        mfu=20.0, steps=100, status="keep", notes="",
        watts=watts, joules_per_token=joules_per_token, total_energy_joules=watts * 300,
        **kwargs,
    )


class TestMemoryPressure:
    def test_memory_growth_detected(self):
        drift = HardwareDrift(HardwareConfig(memory_trend_window=10, memory_growth_pct=10.0))
        # Memory growing from 8 to 12 GB
        results = [_make_result(i, peak_mem_gb=8.0 + i * 0.4) for i in range(12)]
        alerts = drift.check(results)
        memory = [a for a in alerts if a.category == "memory"]
        assert len(memory) > 0

    def test_stable_memory_no_alert(self):
        drift = HardwareDrift(HardwareConfig(memory_trend_window=10))
        results = [_make_result(i, peak_mem_gb=8.0) for i in range(12)]
        alerts = drift.check(results)
        memory = [a for a in alerts if a.category == "memory"]
        assert len(memory) == 0


class TestThermalThrottling:
    def test_throttling_detected(self):
        drift = HardwareDrift(HardwareConfig(thermal_toksec_drop_pct=15.0))
        # Normal operation
        results = [_make_result(i, watts=300, tok_sec=10000) for i in range(6)]
        # Sudden drop in both power and throughput
        results.append(_make_result(6, watts=200, tok_sec=6000))
        alerts = drift.check(results)
        thermal = [a for a in alerts if a.category == "thermal"]
        assert len(thermal) > 0


class TestPowerEfficiency:
    def test_efficiency_drop_without_bpb_improvement(self):
        drift = HardwareDrift(HardwareConfig(
            power_efficiency_window=10,
            power_efficiency_drop_pct=20.0,
        ))
        # First 5: efficient
        results = [_make_result(i, joules_per_token=0.001, val_bpb=1.3) for i in range(5)]
        # Next 5: much less efficient, no bpb improvement
        results += [_make_result(i + 5, joules_per_token=0.002, val_bpb=1.3) for i in range(5)]
        alerts = drift.check(results)
        power = [a for a in alerts if a.category == "power"]
        assert len(power) > 0
