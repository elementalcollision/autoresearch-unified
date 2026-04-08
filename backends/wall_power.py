"""Optional wall-power measurement via MLCommons power-dev server.

Provides a simplified client that coordinates with a running MLCommons
power-dev server (PTDaemon) during training to capture total system AC
power from calibrated Yokogawa meters.

Graceful degradation: if the server is unreachable or not configured,
all methods are no-ops returning zeros. Training never crashes.

Controlled by environment variables:
    AUTORESEARCH_WALL_POWER=1           Enable wall-power measurement
    AUTORESEARCH_WALL_POWER_HOST=host   Server hostname (default: localhost)
    AUTORESEARCH_WALL_POWER_PORT=port   Server port (default: 4950)
"""

import csv
import io
import logging
import os
import socket
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WallPowerResult:
    """Wall-power measurement results from MLCommons server."""
    avg_watts: float = 0.0
    total_joules: float = 0.0
    samples: list[float] = None
    timestamps: list[float] = None
    source: str = ""

    def __post_init__(self):
        if self.samples is None:
            self.samples = []
        if self.timestamps is None:
            self.timestamps = []


class WallPowerAdapter:
    """Sidecar adapter for MLCommons power-dev wall-power measurement.

    Usage:
        adapter = WallPowerAdapter()   # no-op if AUTORESEARCH_WALL_POWER != "1"
        adapter.start()
        # ... training loop ...
        adapter.stop()
        result = adapter.get_results()
    """

    # MLCommons power-dev Protocol v3 constants
    _PROTO_MAGIC = "mlcommons/power client v3"
    _CONNECT_TIMEOUT = 5.0
    _RECV_TIMEOUT = 10.0

    def __init__(self):
        self._enabled = os.environ.get("AUTORESEARCH_WALL_POWER") == "1"
        self._host = os.environ.get("AUTORESEARCH_WALL_POWER_HOST", "localhost")
        self._port = int(os.environ.get("AUTORESEARCH_WALL_POWER_PORT", "4950"))
        self._socket: socket.socket | None = None
        self._connected = False
        self._start_time = 0.0
        self._result = WallPowerResult()
        self._raw_log = ""

    def start(self) -> None:
        """Connect to power-dev server and signal measurement start.

        No-op if not enabled or server unreachable.
        """
        if not self._enabled:
            return

        try:
            self._connect()
            if not self._connected:
                return
            # Send session start + ranging/testing commands
            self._send("new session mlpower_autoresearch")
            self._send("start ranging")
            # Brief pause for ranging phase
            time.sleep(0.5)
            self._send("stop ranging")
            self._send("start testing")
            self._start_time = time.time()
            logger.info("Wall-power measurement started via MLCommons server at %s:%d",
                        self._host, self._port)
        except Exception as e:
            logger.warning("Wall-power start failed (degrading gracefully): %s", e)
            self._cleanup()

    def stop(self) -> None:
        """Signal measurement end and retrieve power log.

        No-op if not connected.
        """
        if not self._connected:
            return

        try:
            elapsed = time.time() - self._start_time
            self._send("stop testing")

            # Request power log data
            response = self._send("get power log")
            if response:
                self._raw_log = response
                self._parse_power_log(response, elapsed)

            self._send("end session")
            logger.info("Wall-power measurement stopped. avg=%.1fW over %.1fs",
                        self._result.avg_watts, elapsed)
        except Exception as e:
            logger.warning("Wall-power stop failed (degrading gracefully): %s", e)
        finally:
            self._cleanup()

    def get_results(self) -> WallPowerResult:
        """Return wall-power measurement results (zeros if unavailable)."""
        return self._result

    # ------------------------------------------------------------------
    # Protocol communication
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Establish TCP connection to MLCommons power-dev server."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self._CONNECT_TIMEOUT)
            sock.connect((self._host, self._port))
            sock.settimeout(self._RECV_TIMEOUT)

            # Send protocol identification
            sock.sendall((self._PROTO_MAGIC + "\n").encode())
            reply = self._recv(sock)
            if reply and "v3" in reply.lower():
                self._socket = sock
                self._connected = True
                logger.debug("Connected to MLCommons power-dev server")
            else:
                logger.warning("Unexpected protocol response: %s", reply)
                sock.close()
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            logger.info("MLCommons power-dev server not available at %s:%d (%s)",
                        self._host, self._port, e)

    def _send(self, command: str) -> str:
        """Send command and return reply."""
        if not self._socket:
            return ""
        try:
            self._socket.sendall((command + "\n").encode())
            return self._recv(self._socket)
        except (socket.timeout, OSError) as e:
            logger.warning("Communication error: %s", e)
            return ""

    @staticmethod
    def _recv(sock: socket.socket) -> str:
        """Receive a newline-terminated response."""
        data = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
                if b"\n" in data:
                    break
            except socket.timeout:
                break
        return data.decode(errors="replace").strip()

    def _cleanup(self) -> None:
        """Close socket and reset state."""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
        self._socket = None
        self._connected = False

    # ------------------------------------------------------------------
    # Power log parsing
    # ------------------------------------------------------------------

    def _parse_power_log(self, log_data: str, elapsed_seconds: float) -> None:
        """Parse MLCommons spl.txt-format power log into results.

        Expected CSV format: timestamp,watts,volts,amps,pf,...
        """
        samples = []
        timestamps = []

        try:
            reader = csv.reader(io.StringIO(log_data))
            for row in reader:
                if len(row) < 2:
                    continue
                try:
                    ts = float(row[0])
                    watts = float(row[1])
                    if watts > 0:
                        timestamps.append(ts)
                        samples.append(watts)
                except (ValueError, IndexError):
                    continue
        except Exception:
            pass

        if samples:
            avg_watts = sum(samples) / len(samples)
            total_joules = avg_watts * elapsed_seconds
            self._result = WallPowerResult(
                avg_watts=avg_watts,
                total_joules=total_joules,
                samples=samples,
                timestamps=timestamps,
                source=f"mlcommons_yokogawa@{self._host}:{self._port}",
            )
        else:
            self._result = WallPowerResult()
