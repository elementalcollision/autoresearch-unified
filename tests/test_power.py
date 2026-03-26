"""Tests for backends.power -- cross-platform power monitoring."""
import time
import pytest
from backends.power import PowerMonitor


class TestPowerMonitorGracefulDegradation:
    """Power monitoring must never crash training -- graceful degradation."""

    def test_unavailable_platform_returns_zeros(self):
        """When no power API is available, stop() returns (0.0, 0.0)."""
        monitor = PowerMonitor(backend="unknown_platform")
        monitor.start()
        time.sleep(0.05)
        avg_watts, total_joules = monitor.stop(training_seconds=100.0)
        assert avg_watts == 0.0
        assert total_joules == 0.0

    def test_start_stop_without_samples(self):
        """start()/stop() is safe even if no samples are collected."""
        monitor = PowerMonitor(backend="unknown_platform")
        monitor.start()
        avg_watts, total_joules = monitor.stop(training_seconds=0.0)
        assert avg_watts == 0.0
        assert total_joules == 0.0

    def test_stop_without_start_is_safe(self):
        """stop() without start() returns zeros gracefully."""
        monitor = PowerMonitor(backend="unknown_platform")
        avg_watts, total_joules = monitor.stop(training_seconds=100.0)
        assert avg_watts == 0.0
        assert total_joules == 0.0

    def test_double_stop_is_safe(self):
        """Calling stop() twice doesn't raise."""
        monitor = PowerMonitor(backend="unknown_platform")
        monitor.start()
        monitor.stop(training_seconds=10.0)
        avg_watts, total_joules = monitor.stop(training_seconds=10.0)
        assert avg_watts == 0.0
        assert total_joules == 0.0


class TestPowerMonitorCalculation:
    """Test that energy calculations are correct."""

    def test_joules_calculation(self):
        """total_joules = avg_watts * training_seconds."""
        monitor = PowerMonitor(backend="cuda")
        # Inject a fake sampler that always returns 200W
        monitor._sampler = lambda: 200.0
        monitor.start()
        time.sleep(0.15)  # collect a few samples
        avg_watts, total_joules = monitor.stop(training_seconds=300.0)
        assert avg_watts == pytest.approx(200.0, abs=5.0)
        assert total_joules == pytest.approx(avg_watts * 300.0, rel=0.01)

    def test_zero_training_seconds(self):
        """Zero training seconds should give zero joules."""
        monitor = PowerMonitor(backend="cuda")
        monitor._sampler = lambda: 350.0
        monitor.start()
        time.sleep(0.1)
        avg_watts, total_joules = monitor.stop(training_seconds=0.0)
        assert avg_watts > 0  # samples were collected
        assert total_joules == 0.0  # but 0 seconds means 0 joules

    def test_sampler_exception_does_not_crash(self):
        """If the sampler raises, the monitor keeps running."""
        call_count = 0
        def flaky_sampler():
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise RuntimeError("simulated hardware error")
            return 100.0

        monitor = PowerMonitor(backend="cuda")
        monitor._sampler = flaky_sampler
        monitor.start()
        time.sleep(0.3)
        avg_watts, total_joules = monitor.stop(training_seconds=60.0)
        # Should have collected some samples despite errors
        assert avg_watts == pytest.approx(100.0, abs=1.0)
        assert total_joules > 0
