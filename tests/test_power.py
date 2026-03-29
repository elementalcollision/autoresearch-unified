"""Tests for backends.power -- cross-platform power monitoring."""
import subprocess
import time
from unittest.mock import patch, MagicMock

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


# ---------------------------------------------------------------------------
# _parse_powermetrics_gpu
# ---------------------------------------------------------------------------

class TestParsepowermetricsGpu:
    """Test plist parsing for Apple Silicon powermetrics output."""

    SAMPLE_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>gpu</key>
    <dict>
        <key>gpu_energy</key>
        <integer>11235</integer>
        <key>gpu_busy_ratio</key>
        <real>0.42</real>
    </dict>
</dict>
</plist>"""

    SAMPLE_PLIST_GPU_POWER = """\
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0">
<dict>
    <key>gpu</key>
    <dict>
        <key>gpu_power</key>
        <real>8500.0</real>
    </dict>
</dict>
</plist>"""

    def test_parses_gpu_energy(self):
        """gpu_energy in millijoules → watts (at 1s interval)."""
        watts = PowerMonitor._parse_powermetrics_gpu(self.SAMPLE_PLIST)
        assert watts == pytest.approx(11.235, rel=0.01)

    def test_parses_gpu_power_fallback(self):
        """Falls back to gpu_power key if gpu_energy is absent."""
        watts = PowerMonitor._parse_powermetrics_gpu(self.SAMPLE_PLIST_GPU_POWER)
        assert watts == pytest.approx(8.5, rel=0.01)

    def test_empty_output_returns_zero(self):
        assert PowerMonitor._parse_powermetrics_gpu("") == 0.0

    def test_malformed_plist_returns_zero(self):
        assert PowerMonitor._parse_powermetrics_gpu("<plist>garbage</plist>") == 0.0

    def test_no_gpu_keys_returns_zero(self):
        plist = """\
<plist version="1.0"><dict>
<key>cpu_energy</key><integer>5000</integer>
</dict></plist>"""
        assert PowerMonitor._parse_powermetrics_gpu(plist) == 0.0


# ---------------------------------------------------------------------------
# _try_metal_sampler auto-detection
# ---------------------------------------------------------------------------

class TestMetalSamplerAutoDetect:
    """Test that _try_metal_sampler auto-detects passwordless sudo."""

    GOOD_PLIST = """\
<?xml version="1.0" encoding="UTF-8"?>
<plist version="1.0"><dict>
<key>gpu</key><dict>
<key>gpu_energy</key><integer>15000</integer>
</dict></dict></plist>"""

    @patch("subprocess.run")
    def test_returns_sampler_when_sudo_works(self, mock_run):
        """Returns a callable sampler when sudo powermetrics succeeds."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self.GOOD_PLIST
        )
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        sampler = monitor._try_metal_sampler()
        assert sampler is not None
        assert callable(sampler)

    @patch("subprocess.run")
    def test_returns_none_when_sudo_fails(self, mock_run):
        """Returns None when sudo requires a password (returncode != 0)."""
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="sudo: a password is required"
        )
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        assert monitor._try_metal_sampler() is None

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_returns_none_when_powermetrics_missing(self, mock_run):
        """Returns None on Linux/non-macOS where powermetrics doesn't exist."""
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        assert monitor._try_metal_sampler() is None

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="sudo", timeout=10))
    def test_returns_none_on_timeout(self, mock_run):
        """Returns None if powermetrics times out (e.g. sudo hangs)."""
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        assert monitor._try_metal_sampler() is None

    @patch("subprocess.run")
    def test_returns_none_when_gpu_energy_zero(self, mock_run):
        """Returns None if powermetrics runs but reports 0 GPU energy."""
        zero_plist = """\
<plist version="1.0"><dict>
<key>gpu</key><dict>
<key>gpu_energy</key><integer>0</integer>
</dict></dict></plist>"""
        mock_run.return_value = MagicMock(returncode=0, stdout=zero_plist)
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        assert monitor._try_metal_sampler() is None

    @patch("subprocess.run")
    def test_sampler_callable_invokes_powermetrics(self, mock_run):
        """The returned sampler calls powermetrics and parses the result."""
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self.GOOD_PLIST
        )
        monitor = PowerMonitor.__new__(PowerMonitor)
        monitor._poll_interval = 1.0
        sampler = monitor._try_metal_sampler()
        # Reset mock to track the sampler invocation
        mock_run.reset_mock()
        mock_run.return_value = MagicMock(
            returncode=0, stdout=self.GOOD_PLIST
        )
        watts = sampler()
        assert watts == pytest.approx(15.0, rel=0.01)
        mock_run.assert_called_once()
