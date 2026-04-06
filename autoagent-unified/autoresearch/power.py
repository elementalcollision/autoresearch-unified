# Ported from autoresearch-unified/backends/power.py (MIT)
"""Cross-platform power monitoring for energy instrumentation.

Supports NVIDIA (nvidia-ml-py / nvidia-smi), AMD ROCm (amdsmi / rocm-smi),
Apple Silicon (powermetrics), and Intel Gaudi (hl-smi).

Graceful degradation: if no power API is available, start()/stop() are no-ops.
"""

import json
import re
import subprocess
import threading
import time


class PowerMonitor:
    """Background power sampler for energy-per-token instrumentation."""

    def __init__(self, backend: str, poll_interval: float = 1.0):
        self._backend = backend
        self._poll_interval = poll_interval
        self._samples: list[float] = []
        self._running = False
        self._thread: threading.Thread | None = None
        self._sampler = self._detect_sampler()

    def start(self) -> None:
        if self._sampler is None:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self, training_seconds: float) -> tuple[float, float]:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if not self._samples:
            return 0.0, 0.0
        avg_watts = sum(self._samples) / len(self._samples)
        total_joules = avg_watts * training_seconds
        return avg_watts, total_joules

    def _detect_sampler(self):
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

    def _try_cuda_sampler(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            if mw > 0:
                def _sample():
                    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                return _sample
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                val = float(result.stdout.strip().split("\n")[0])
                if val > 0:
                    def _sample_smi():
                        r = subprocess.run(
                            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                        )
                        return float(r.stdout.strip().split("\n")[0])
                    return _sample_smi
        except Exception:
            pass
        return None

    def _try_rocm_sampler(self):
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if handles:
                handle = handles[0]
                info = amdsmi.amdsmi_get_power_info(handle)
                power = float(info.get("average_socket_power", 0))
                if power > 0:
                    def _sample():
                        pi = amdsmi.amdsmi_get_power_info(handle)
                        return float(pi.get("average_socket_power", 0))
                    return _sample
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                power = self._parse_rocm_smi_power(data)
                if power > 0:
                    def _sample_rocm():
                        r = subprocess.run(
                            ["rocm-smi", "--showpower", "--json"],
                            capture_output=True, text=True, timeout=5,
                        )
                        return self._parse_rocm_smi_power(json.loads(r.stdout))
                    return _sample_rocm
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_rocm_smi_power(data: dict) -> float:
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
        m = re.search(r"(\d+\.?\d*)\s*W", json.dumps(data))
        if m:
            return float(m.group(1))
        return 0.0

    def _try_metal_sampler(self):
        try:
            result = subprocess.run(
                ["sudo", "-n", "powermetrics",
                 "--samplers", "gpu_power", "-i", "1000", "-n", "1", "-f", "plist"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                power = self._parse_powermetrics_gpu(result.stdout)
                if power > 0:
                    def _sample():
                        r = subprocess.run(
                            ["sudo", "-n", "powermetrics",
                             "--samplers", "gpu_power", "-i", "1000", "-n", "1", "-f", "plist"],
                            capture_output=True, text=True, timeout=10,
                        )
                        return self._parse_powermetrics_gpu(r.stdout)
                    return _sample
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_powermetrics_gpu(output: str) -> float:
        m = re.search(r"<key>gpu_energy</key>\s*<integer>(\d+)</integer>", output)
        if m:
            return float(m.group(1)) / 1000.0
        m = re.search(r"<key>gpu_power</key>\s*<(?:integer|real)>([\d.]+)</(?:integer|real)>", output)
        if m:
            return float(m.group(1)) / 1000.0
        return 0.0

    def _try_gaudi_sampler(self):
        try:
            result = subprocess.run(
                ["hl-smi", "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                power = self._parse_hlsmi_power(result.stdout)
                if power > 0:
                    def _sample():
                        r = subprocess.run(
                            ["hl-smi", "-q", "-d", "POWER"],
                            capture_output=True, text=True, timeout=10,
                        )
                        return self._parse_hlsmi_power(r.stdout)
                    return _sample
        except Exception:
            pass
        return None

    @staticmethod
    def _parse_hlsmi_power(output: str) -> float:
        m = re.search(r"Power Draw\s*:\s*([\d.]+)\s*W", output)
        if m:
            return float(m.group(1))
        return 0.0

    def _poll_loop(self) -> None:
        while self._running:
            try:
                w = self._sampler()
                if w > 0:
                    self._samples.append(w)
            except Exception:
                pass
            time.sleep(self._poll_interval)
