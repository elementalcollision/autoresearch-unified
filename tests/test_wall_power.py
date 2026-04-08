"""Tests for backends.wall_power and backends.power_report."""

import socket
from unittest.mock import patch, MagicMock

import pytest
from backends.wall_power import WallPowerAdapter, WallPowerResult
from backends.power_report import CombinedPowerReport


# ---------------------------------------------------------------------------
# WallPowerAdapter -- graceful degradation
# ---------------------------------------------------------------------------

class TestWallPowerGracefulDegradation:
    """Wall-power adapter must never crash training."""

    def test_disabled_by_default(self):
        """When AUTORESEARCH_WALL_POWER is not set, all methods are no-ops."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = WallPowerAdapter()
            adapter.start()   # should not connect
            adapter.stop()    # should not fail
            result = adapter.get_results()
            assert result.avg_watts == 0.0
            assert result.total_joules == 0.0

    @patch.dict("os.environ", {"AUTORESEARCH_WALL_POWER": "1"})
    @patch("socket.socket")
    def test_unreachable_server_returns_zeros(self, mock_socket_cls):
        """Unreachable server degrades gracefully."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = ConnectionRefusedError
        mock_socket_cls.return_value = mock_sock

        adapter = WallPowerAdapter()
        adapter.start()
        adapter.stop()
        result = adapter.get_results()
        assert result.avg_watts == 0.0

    @patch.dict("os.environ", {"AUTORESEARCH_WALL_POWER": "1"})
    @patch("socket.socket")
    def test_timeout_returns_zeros(self, mock_socket_cls):
        """Socket timeout degrades gracefully."""
        mock_sock = MagicMock()
        mock_sock.connect.side_effect = socket.timeout
        mock_socket_cls.return_value = mock_sock

        adapter = WallPowerAdapter()
        adapter.start()
        adapter.stop()
        result = adapter.get_results()
        assert result.avg_watts == 0.0

    def test_stop_without_start(self):
        """stop() without start() is safe."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = WallPowerAdapter()
            adapter.stop()  # should not raise

    def test_double_stop(self):
        """Calling stop() twice is safe."""
        with patch.dict("os.environ", {}, clear=True):
            adapter = WallPowerAdapter()
            adapter.stop()
            adapter.stop()


# ---------------------------------------------------------------------------
# WallPowerAdapter -- protocol
# ---------------------------------------------------------------------------

class TestWallPowerProtocol:
    """Test MLCommons Protocol v3 communication."""

    @patch.dict("os.environ", {"AUTORESEARCH_WALL_POWER": "1"})
    @patch("socket.socket")
    def test_sends_protocol_magic(self, mock_socket_cls):
        """Client sends protocol identification on connect."""
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b"mlcommons/power server v3\n"
        mock_socket_cls.return_value = mock_sock

        adapter = WallPowerAdapter()
        adapter._connect()

        # First sendall should be protocol magic
        calls = mock_sock.sendall.call_args_list
        assert len(calls) >= 1
        assert b"mlcommons/power client v3" in calls[0][0][0]

    @patch.dict("os.environ", {"AUTORESEARCH_WALL_POWER": "1"})
    @patch("socket.socket")
    def test_rejects_wrong_protocol(self, mock_socket_cls):
        """Does not connect if server responds with wrong protocol."""
        mock_sock = MagicMock()
        mock_sock.recv.return_value = b"unknown protocol\n"
        mock_socket_cls.return_value = mock_sock

        adapter = WallPowerAdapter()
        adapter._connect()
        assert adapter._connected is False


# ---------------------------------------------------------------------------
# WallPowerAdapter -- log parsing
# ---------------------------------------------------------------------------

class TestWallPowerLogParsing:
    """Test spl.txt CSV power log parsing."""

    def test_parses_csv_log(self):
        """Correctly parses timestamp,watts CSV format."""
        adapter = WallPowerAdapter.__new__(WallPowerAdapter)
        adapter._result = WallPowerResult()
        adapter._host = "localhost"
        adapter._port = 4950

        log_data = "1000.0,850.3,120.0,7.1,0.99\n1001.0,855.1,120.1,7.1,0.99\n1002.0,848.7,119.9,7.1,0.99\n"
        adapter._parse_power_log(log_data, elapsed_seconds=300.0)

        result = adapter._result
        assert result.avg_watts == pytest.approx(851.37, abs=0.1)
        assert result.total_joules == pytest.approx(851.37 * 300.0, rel=0.01)
        assert len(result.samples) == 3
        assert len(result.timestamps) == 3

    def test_empty_log_returns_zeros(self):
        """Empty log data results in zero watts."""
        adapter = WallPowerAdapter.__new__(WallPowerAdapter)
        adapter._result = WallPowerResult()
        adapter._host = "localhost"
        adapter._port = 4950
        adapter._parse_power_log("", elapsed_seconds=300.0)
        assert adapter._result.avg_watts == 0.0

    def test_malformed_log_skips_bad_rows(self):
        """Malformed rows are skipped without error."""
        adapter = WallPowerAdapter.__new__(WallPowerAdapter)
        adapter._result = WallPowerResult()
        adapter._host = "localhost"
        adapter._port = 4950
        log_data = "bad,data\n1000.0,500.0\nmore,garbage\n"
        adapter._parse_power_log(log_data, elapsed_seconds=60.0)
        assert adapter._result.avg_watts == pytest.approx(500.0)
        assert len(adapter._result.samples) == 1


# ---------------------------------------------------------------------------
# CombinedPowerReport
# ---------------------------------------------------------------------------

class TestCombinedPowerReport:
    """Test combined report generation and derived metrics."""

    def test_gpu_only(self):
        """When no wall data, report is gpu_only."""
        report = CombinedPowerReport.from_sources(
            gpu_watts=470.0,
            gpu_joules=141000.0,
            wall_data=None,
            training_seconds=300.0,
            total_tokens=600_000_000,
        )
        assert report.measurement_quality == "gpu_only"
        assert report.gpu_avg_watts == 470.0
        assert report.gpu_total_joules == 141000.0
        assert report.gpu_joules_per_token == pytest.approx(141000.0 / 600_000_000)
        assert report.wall_avg_watts == 0.0
        assert report.gpu_power_fraction == 0.0

    def test_combined_metrics(self):
        """When both sources available, derived metrics are computed."""
        wall = WallPowerResult(avg_watts=850.0, total_joules=255000.0)
        report = CombinedPowerReport.from_sources(
            gpu_watts=470.0,
            gpu_joules=141000.0,
            wall_data=wall,
            training_seconds=300.0,
            total_tokens=600_000_000,
        )
        assert report.measurement_quality == "combined"
        assert report.gpu_power_fraction == pytest.approx(470.0 / 850.0, rel=0.001)
        assert report.overhead_watts == pytest.approx(380.0)
        assert report.pue_estimate == pytest.approx(850.0 / 470.0, rel=0.001)
        assert report.wall_avg_watts == 850.0
        assert report.wall_total_joules == pytest.approx(850.0 * 300.0)

    def test_wall_only(self):
        """When GPU power is 0 but wall is available."""
        wall = WallPowerResult(avg_watts=850.0)
        report = CombinedPowerReport.from_sources(
            gpu_watts=0.0,
            gpu_joules=0.0,
            wall_data=wall,
            training_seconds=300.0,
            total_tokens=600_000_000,
        )
        assert report.measurement_quality == "wall_only"
        assert report.wall_avg_watts == 850.0
        assert report.gpu_power_fraction == 0.0

    def test_zero_tokens(self):
        """Zero tokens doesn't crash (joules_per_token = 0)."""
        report = CombinedPowerReport.from_sources(
            gpu_watts=470.0,
            gpu_joules=141000.0,
            wall_data=None,
            training_seconds=300.0,
            total_tokens=0,
        )
        assert report.gpu_joules_per_token == 0.0

    def test_empty_wall_result(self):
        """WallPowerResult with 0 watts is treated as unavailable."""
        wall = WallPowerResult(avg_watts=0.0)
        report = CombinedPowerReport.from_sources(
            gpu_watts=470.0,
            gpu_joules=141000.0,
            wall_data=wall,
            training_seconds=300.0,
            total_tokens=600_000_000,
        )
        assert report.measurement_quality == "gpu_only"
