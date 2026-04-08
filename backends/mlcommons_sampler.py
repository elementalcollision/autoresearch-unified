"""MLCommons-compatible GPU power sampler for the power-dev framework.

Conforms to the MLCommons power_meter_sampling sampler interface:
  - close()       -- cleanup resources
  - get_titles()  -- return CSV column headers
  - get_values()  -- return current power readings

This sampler provides per-GPU software-based power readings that complement
MLCommons' external Yokogawa wall-power measurements. It supports NVIDIA
(pynvml/nvidia-smi), AMD ROCm (amdsmi/rocm-smi), and Intel Gaudi (hl-smi).

Standalone: no autoresearch dependencies. Can be contributed to
mlcommons/power-dev as a plugin in power_meter_sampling/samplers/.

Apache 2.0 License (compatible with MLCommons power-dev).
"""

import json
import re
import subprocess


class GPUPowerSampler:
    """Per-GPU software power sampler for MLCommons power-dev integration.

    Usage with MLCommons power_meter_sampling framework:
        sampler = GPUPowerSampler()
        titles = sampler.get_titles()   # ("GPU0_Power_W", "GPU1_Power_W", ...)
        values = sampler.get_values()   # (245.3, 238.7, ...)
        sampler.close()
    """

    def __init__(self):
        self._platform = None
        self._handles = []
        self._cleanup = None
        self._num_gpus = 0
        self._detect()

    def close(self):
        """Release hardware handles."""
        if self._cleanup:
            try:
                self._cleanup()
            except Exception:
                pass
        self._handles = []
        self._num_gpus = 0
        self._platform = None

    def get_titles(self) -> tuple:
        """Return CSV column headers for each GPU."""
        return tuple(f"GPU{i}_Power_W" for i in range(self._num_gpus))

    def get_values(self) -> tuple:
        """Return current power readings (watts) for each GPU."""
        if self._platform == "pynvml":
            return self._read_pynvml()
        elif self._platform == "nvidia-smi":
            return self._read_nvidia_smi()
        elif self._platform == "amdsmi":
            return self._read_amdsmi()
        elif self._platform == "rocm-smi":
            return self._read_rocm_smi()
        elif self._platform == "hl-smi":
            return self._read_hlsmi()
        return tuple(0.0 for _ in range(self._num_gpus))

    # ------------------------------------------------------------------
    # Platform detection (multi-GPU)
    # ------------------------------------------------------------------

    def _detect(self):
        """Probe hardware and configure for multi-GPU reading."""
        if self._try_pynvml():
            return
        if self._try_nvidia_smi():
            return
        if self._try_amdsmi():
            return
        if self._try_rocm_smi():
            return
        if self._try_hlsmi():
            return
        # No GPU power API available — sampler produces empty columns
        self._num_gpus = 0

    def _try_pynvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                return False
            handles = []
            for i in range(count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                mw = pynvml.nvmlDeviceGetPowerUsage(h)
                if mw <= 0:
                    return False
                handles.append(h)
            self._handles = handles
            self._num_gpus = count
            self._platform = "pynvml"
            self._cleanup = lambda: pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    def _try_nvidia_smi(self) -> bool:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            values = [float(l) for l in lines]
            if not values or any(v <= 0 for v in values):
                return False
            self._num_gpus = len(values)
            self._platform = "nvidia-smi"
            return True
        except Exception:
            return False

    def _try_amdsmi(self) -> bool:
        try:
            import amdsmi
            amdsmi.amdsmi_init()
            handles = amdsmi.amdsmi_get_processor_handles()
            if not handles:
                return False
            # Verify all handles can report power
            for h in handles:
                info = amdsmi.amdsmi_get_power_info(h)
                if float(info.get("average_socket_power", 0)) <= 0:
                    return False
            self._handles = list(handles)
            self._num_gpus = len(handles)
            self._platform = "amdsmi"
            self._cleanup = lambda: amdsmi.amdsmi_shut_down()
            return True
        except Exception:
            return False

    def _try_rocm_smi(self) -> bool:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return False
            data = json.loads(result.stdout)
            powers = self._parse_rocm_smi_multi(data)
            if not powers or any(p <= 0 for p in powers):
                return False
            self._num_gpus = len(powers)
            self._platform = "rocm-smi"
            return True
        except Exception:
            return False

    def _try_hlsmi(self) -> bool:
        try:
            result = subprocess.run(
                ["hl-smi", "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return False
            powers = re.findall(r"Power Draw\s*:\s*([\d.]+)\s*W", result.stdout)
            if not powers:
                return False
            values = [float(p) for p in powers]
            if any(v <= 0 for v in values):
                return False
            self._num_gpus = len(values)
            self._platform = "hl-smi"
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Readers
    # ------------------------------------------------------------------

    def _read_pynvml(self) -> tuple:
        import pynvml
        values = []
        for h in self._handles:
            try:
                values.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
            except Exception:
                values.append(0.0)
        return tuple(values)

    def _read_nvidia_smi(self) -> tuple:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            return tuple(float(l) for l in lines[:self._num_gpus])
        except Exception:
            return tuple(0.0 for _ in range(self._num_gpus))

    def _read_amdsmi(self) -> tuple:
        import amdsmi
        values = []
        for h in self._handles:
            try:
                info = amdsmi.amdsmi_get_power_info(h)
                values.append(float(info.get("average_socket_power", 0)))
            except Exception:
                values.append(0.0)
        return tuple(values)

    def _read_rocm_smi(self) -> tuple:
        try:
            result = subprocess.run(
                ["rocm-smi", "--showpower", "--json"],
                capture_output=True, text=True, timeout=5,
            )
            data = json.loads(result.stdout)
            powers = self._parse_rocm_smi_multi(data)
            return tuple(powers[:self._num_gpus])
        except Exception:
            return tuple(0.0 for _ in range(self._num_gpus))

    def _read_hlsmi(self) -> tuple:
        try:
            result = subprocess.run(
                ["hl-smi", "-q", "-d", "POWER"],
                capture_output=True, text=True, timeout=10,
            )
            matches = re.findall(r"Power Draw\s*:\s*([\d.]+)\s*W", result.stdout)
            values = [float(m) for m in matches[:self._num_gpus]]
            while len(values) < self._num_gpus:
                values.append(0.0)
            return tuple(values)
        except Exception:
            return tuple(0.0 for _ in range(self._num_gpus))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_rocm_smi_multi(data: dict) -> list[float]:
        """Extract per-GPU power from rocm-smi JSON (multi-GPU)."""
        powers = []
        for key in sorted(data.keys()):
            val = data[key]
            if not isinstance(val, dict):
                continue
            power = 0.0
            for subkey, subval in val.items():
                if "power" in subkey.lower() and "w" in subkey.lower():
                    try:
                        power = float(str(subval).replace("W", "").strip())
                        break
                    except (ValueError, TypeError):
                        pass
                if isinstance(subval, str):
                    m = re.search(r"(\d+\.?\d*)\s*W", subval)
                    if m:
                        power = float(m.group(1))
                        break
            if power > 0:
                powers.append(power)
        return powers
