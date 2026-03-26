"""Cross-platform power monitoring for energy instrumentation.

Provides a background thread that polls GPU/accelerator power draw during
training. Supports NVIDIA (pynvml / nvidia-smi), AMD ROCm (amdsmi / rocm-smi),
Apple Silicon (ioreg), and Intel Gaudi (hl-smi).

Graceful degradation: if no power API is available, start()/stop() are no-ops
and stop() returns (0.0, 0.0). Training never crashes due to power monitoring.
"""

import json
import re
import subprocess
import threading
import time


class PowerMonitor:
    """Background power sampler for energy-per-token instrumentation.

    Usage:
        monitor = PowerMonitor(backend="cuda")
        monitor.start()
        # ... training loop ...
        avg_watts, total_joules = monitor.stop(training_seconds=total_training_time)
        joules_per_token = total_joules / total_tokens if total_tokens > 0 else 0.0
    """

    def __init__(self, backend: str, poll_interval: float = 1.0):
        self._backend = backend
        self._poll_interval = poll_interval
        self._samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._sampler = self._detect_sampler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin background power sampling. No-op if no sampler available."""
        if self._sampler is None:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, training_seconds: float) -> tuple[float, float]:
        """Stop sampling and return (avg_watts, total_joules).

        total_joules = avg_watts * training_seconds.
        Returns (0.0, 0.0) if no samples were collected.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self._samples:
            return 0.0, 0.0
        avg_watts = sum(self._samples) / len(self._samples)
        total_joules = avg_watts * training_seconds
        return avg_watts, total_joules

    # ------------------------------------------------------------------
    # Sampler detection
    # ------------------------------------------------------------------

    def _detect_sampler(self):
        """Probe hardware and return a callable sampler, or None."""
        if self._backend in ("cuda",):
            sampler = self._try_cuda_sampler()
            if sampler:
                return sampler

        if self._backend in ("rocm", "rocm7"):
            sampler = self._try_rocm_sampler()
            if sampler:
                return sampler

        if self._backend in ("mlx", "mps"):
            sampler = self._try_metal_sampler()
            if sampler:
                return sampler

        if self._backend in ("hpu",):
            sampler = self._try_gaudi_sampler()
            if sampler:
                return sampler

        return None

    # ------------------------------------------------------------------
    # Platform-specific sampler factories
    # ------------------------------------------------------------------

    def _try_cuda_sampler(self):
        """Try pynvml first, fall back to nvidia-smi."""
        # Try pynvml (fast, in-process)
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # Test read
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            if mw > 0:
                def _sample_pynvml():
                    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                return _sample_pynvml
        except Exception:
            pass

        # Fallback: nvidia-smi subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                val = float(result.stdout.strip().split("\n")[0])
                if val > 0:
                    def _sample_nvidia_smi():
                        r = subprocess.run(
                            ["nvidia-smi", "--query-gpu=power.draw",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                        )
                        return float(r.stdout.strip().split("\n")[0])
                    return _sample_nvidia_smi
        except Exception:
            pass

        return None

    def _try_rocm_sampler(self):
        """Try amdsmi first, fall back to rocm-smi."""
        # Try amdsmi (ROCm 5.7+)
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if handles:
                handle = handles[0]
                info = amdsmi.amdsmi_get_power_info(handle)
                power = float(info.get("average_socket_power", 0))
                if power > 0:
                    def _sample_amdsmi():
                        pi = amdsmi.amdsmi_get_power_info(handle)
                        return float(pi.get("average_socket_power", 0))
                    return _sample_amdsmi
        except Exception:
            pass

        # Fallback: rocm-smi subprocess
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Extract power from JSON (key varies by ROCm version)
                power = self._parse_rocm_smi_power(data)
                if power > 0:
                    def _sample_rocm_smi():
                        r = subprocess.run(
                            ["rocm-smi", "--showpower", "--json"],
                            capture_output=True, text=True, timeout=5,
                        )
                        d = json.loads(r.stdout)
                        return self._parse_rocm_smi_power(d)
                    return _sample_rocm_smi
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_rocm_smi_power(data: dict) -> float:
        """Extract power in watts from rocm-smi JSON output."""
        # rocm-smi JSON structure varies by version; try common patterns
        for key, val in data.items():
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if "power" in subkey.lower() and "w" in subkey.lower():
                        try:
                            return float(str(subval).replace("W", "").strip())
                        except (ValueError, TypeError):
                            pass
                    if isinstance(subval, str):
                        m = re.search(r"(\d+\.?\d*)\s*W", subval)
                        if m:
                            return float(m.group(1))
        # Last resort: search entire JSON string
        m = re.search(r"(\d+\.?\d*)\s*W", json.dumps(data))
        if m:
            return float(m.group(1))
        return 0.0

    def _try_metal_sampler(self):
        """Try ioreg for Apple Silicon GPU power (no sudo needed)."""
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOReport"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                # Look for GPU-related power entries
                power = self._parse_ioreg_gpu_power(result.stdout)
                if power > 0:
                    def _sample_ioreg():
                        r = subprocess.run(
                            ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOReport"],
                            capture_output=True, text=True, timeout=5,
                        )
                        return self._parse_ioreg_gpu_power(r.stdout)
                    return _sample_ioreg
        except Exception:
            pass

        # Alternative ioreg query for GPU power
        try:
            result = subprocess.run(
                ["ioreg", "-r", "-n", "AGXAccelerator", "-w", "0"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                # On some Apple Silicon chips, ioreg reports are different
                power = self._parse_agx_power(result.stdout)
                if power > 0:
                    def _sample_agx():
                        r = subprocess.run(
                            ["ioreg", "-r", "-n", "AGXAccelerator", "-w", "0"],
                            capture_output=True, text=True, timeout=5,
                        )
                        return self._parse_agx_power(r.stdout)
                    return _sample_agx
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_ioreg_gpu_power(output: str) -> float:
        """Parse GPU power from ioreg output (milliwatts -> watts)."""
        # Look for GPU power patterns in ioreg output
        for pattern in [
            r'"GPU Power"\s*=\s*(\d+)',
            r'"gpu-power"\s*=\s*(\d+)',
            r'"GPUPower"\s*=\s*(\d+)',
        ]:
            m = re.search(pattern, output, re.IGNORECASE)
            if m:
                mw = float(m.group(1))
                return mw / 1000.0  # milliwatts to watts
        return 0.0

    @staticmethod
    def _parse_agx_power(output: str) -> float:
        """Parse power from AGXAccelerator ioreg node."""
        m = re.search(r'"device-power"\s*=\s*(\d+)', output)
        if m:
            return float(m.group(1)) / 1000.0
        return 0.0

    def _try_gaudi_sampler(self):
        """Try hl-smi for Intel Gaudi power."""
        try:
            result = subprocess.run(
                ["hl-smi", "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                power = self._parse_hlsmi_power(result.stdout)
                if power > 0:
                    def _sample_hlsmi():
                        r = subprocess.run(
                            ["hl-smi", "-q", "-d", "POWER"],
                            capture_output=True, text=True, timeout=10,
                        )
                        return self._parse_hlsmi_power(r.stdout)
                    return _sample_hlsmi
        except Exception:
            pass

        return None

    @staticmethod
    def _parse_hlsmi_power(output: str) -> float:
        """Parse power draw from hl-smi output."""
        m = re.search(r"Power Draw\s*:\s*([\d.]+)\s*W", output)
        if m:
            return float(m.group(1))
        return 0.0

    # ------------------------------------------------------------------
    # Background poll loop
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Background thread: sample power at regular intervals."""
        while self._running:
            try:
                w = self._sampler()
                if w > 0:
                    self._samples.append(w)
            except Exception:
                pass  # Never crash the training script
            time.sleep(self._poll_interval)
