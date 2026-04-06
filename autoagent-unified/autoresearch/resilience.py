# Ported from autoresearch-unified/tui/resilience.py (MIT)
"""Resilience utilities for the experiment runner.

Provides crash-safe file operations, heartbeat monitoring, and signal handling
so overnight runs survive API outages, power losses, and terminal kills.

Components:
  - atomic_write(): Crash-safe file writes (write to .tmp, fsync, rename)
  - validate_results_tsv(): Detect and fix corrupted results files
  - Heartbeat: Status file for external monitors (monitor.py)
  - install_signal_handlers(): Graceful shutdown on SIGTERM/SIGHUP/SIGINT
"""

import json
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Atomic file writer
# ---------------------------------------------------------------------------

def atomic_write(path: str, content: str) -> None:
    """Write content to a file atomically via tmp + rename.

    os.replace() is atomic on both POSIX and Windows — the file is
    either fully written or not changed at all. No corruption from
    mid-write crashes.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def atomic_append(path: str, line: str) -> None:
    """Append a line to a file with fsync for durability.

    Uses direct append + fsync rather than read-all/write-all/rename.
    Direct append is safe because:
    - Single writer (orchestrator is single-threaded)
    - fsync ensures the append hits disk
    - Partial line from a crash is detectable by validate_results_tsv()
    """
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# TSV validation
# ---------------------------------------------------------------------------

EXPECTED_FIELDS = 14
ACCEPTED_FIELD_COUNTS = {10, 11, 14}  # 10=legacy, 11=pre-energy, 14=current

def validate_results_tsv(path: str) -> tuple[bool, list[str]]:
    """Validate a results.tsv file and fix minor corruption.

    Returns (is_valid, list_of_warnings). Fixes issues in-place.
    """
    if not os.path.exists(path):
        return True, []

    warnings = []

    with open(path) as f:
        content = f.read()

    if not content.strip():
        return True, []

    lines = content.split("\n")

    # Check for incomplete trailing line (mid-write crash)
    if content and not content.endswith("\n"):
        last_line = lines[-1]
        fields = last_line.split("\t")
        if len(fields) not in ACCEPTED_FIELD_COUNTS:
            warnings.append(f"Truncated trailing line removed: '{last_line[:60]}...'")
            lines = lines[:-1]
            fixed = "\n".join(lines)
            if not fixed.endswith("\n"):
                fixed += "\n"
            atomic_write(path, fixed)

    # Validate header
    if lines and lines[0].strip():
        header_fields = lines[0].strip().split("\t")
        if header_fields[0] != "exp":
            warnings.append(f"Unexpected header: {lines[0][:60]}")

    # Validate data lines
    for i, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) not in ACCEPTED_FIELD_COUNTS:
            warnings.append(f"Line {i}: expected one of {sorted(ACCEPTED_FIELD_COUNTS)} fields, got {len(fields)}")
            continue
        try:
            float(fields[2])  # val_bpb
            float(fields[3])  # peak_mem_gb
            int(fields[4])    # tok_sec
            float(fields[5])  # mfu
            int(fields[6])    # steps
        except (ValueError, IndexError):
            warnings.append(f"Line {i}: numeric parse error in '{fields[0]}'")

    is_valid = len(warnings) == 0
    return is_valid, warnings


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class Heartbeat:
    """Lightweight status file for external monitors."""

    DEFAULT_PATH = ".runner_status.json"

    def __init__(self, path: str | None = None):
        self._path = path or self.DEFAULT_PATH
        self._started_at = datetime.now().isoformat()
        self._pid = os.getpid()

    def update(
        self,
        experiment: int = 0,
        status: str = "running",
        dataset: str = "",
        best_bpb: float = float("inf"),
        model: str = "",
        total: int = 0,
        kept: int = 0,
        discarded: int = 0,
        crashes: int = 0,
    ) -> None:
        """Write current status to the heartbeat file."""
        data = {
            "pid": self._pid,
            "alive": True,
            "started_at": self._started_at,
            "last_updated": datetime.now().isoformat(),
            "experiment": experiment,
            "status": status,
            "dataset": dataset,
            "model": model,
            "best_bpb": best_bpb if best_bpb < float("inf") else None,
            "total": total,
            "kept": kept,
            "discarded": discarded,
            "crashes": crashes,
        }
        try:
            atomic_write(self._path, json.dumps(data, indent=2) + "\n")
        except Exception:
            pass  # Best-effort, never crash the runner

    def close(self) -> None:
        """Mark the runner as no longer alive."""
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    data = json.load(f)
                data["alive"] = False
                data["last_updated"] = datetime.now().isoformat()
                data["status"] = "stopped"
                atomic_write(self._path, json.dumps(data, indent=2) + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_shutdown_requested = False
_shutdown_time = 0.0


def install_signal_handlers(stop_callback: Callable[[], None]) -> None:
    """Install graceful shutdown handlers for SIGINT, SIGTERM, SIGHUP.

    First signal: set graceful shutdown flag, call stop_callback.
    Second signal within 5s: force exit.
    Must be called from the main thread.
    """
    global _shutdown_requested, _shutdown_time

    def _handler(signum, frame):
        global _shutdown_requested, _shutdown_time
        sig_name = signal.Signals(signum).name

        if _shutdown_requested and (time.time() - _shutdown_time) < 5.0:
            print(f"\n[SIGNAL] Received {sig_name} again — forcing exit", flush=True)
            sys.exit(1)

        _shutdown_requested = True
        _shutdown_time = time.time()
        print(f"\n[SIGNAL] Received {sig_name} — finishing current experiment, then stopping...", flush=True)
        stop_callback()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handler)


# ---------------------------------------------------------------------------
# PID file locking
# ---------------------------------------------------------------------------

_DEFAULT_PIDFILE = Path(__file__).parent.parent / ".suite.pid"


def acquire_pidlock(pidfile: Path = _DEFAULT_PIDFILE) -> bool:
    """Write our PID to the lock file. Returns False if another runner is active."""
    if pidfile.exists():
        try:
            old_pid = int(pidfile.read_text().strip())
            try:
                os.kill(old_pid, 0)
                import platform
                if platform.system() == "Linux":
                    cmdline_path = f"/proc/{old_pid}/cmdline"
                    if os.path.exists(cmdline_path):
                        with open(cmdline_path) as f:
                            cmdline = f.read()
                        if "run_suite" in cmdline or "headless" in cmdline or "dashboard" in cmdline:
                            print(f"\n  ERROR: Another experiment runner is already active (PID {old_pid})")
                            print(f"     Kill it first:  kill {old_pid}")
                            print(f"     Or force:       rm {pidfile} && re-run\n")
                            return False
                else:
                    print(f"\n  ERROR: Another experiment runner may be active (PID {old_pid})")
                    print(f"     Kill it first:  kill {old_pid}")
                    print(f"     Or force:       rm {pidfile} && re-run\n")
                    return False
            except ProcessLookupError:
                pass
            except PermissionError:
                print(f"\n  ERROR: Another experiment runner may be active (PID {old_pid})")
                print(f"     Kill it first:  kill {old_pid}")
                print(f"     Or force:       rm {pidfile} && re-run\n")
                return False
        except (ValueError, OSError):
            pass

    pidfile.write_text(str(os.getpid()))
    return True


def release_pidlock(pidfile: Path = _DEFAULT_PIDFILE):
    """Remove the PID lock file if it still contains our PID."""
    try:
        if pidfile.exists():
            stored = int(pidfile.read_text().strip())
            if stored == os.getpid():
                pidfile.unlink()
    except (ValueError, OSError):
        pass
