"""Tests for backends.mlcommons_sampler -- MLCommons GPU power sampler."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest
from backends.mlcommons_sampler import GPUPowerSampler


class TestGPUSamplerInterface:
    """Verify the sampler conforms to the MLCommons interface contract."""

    def test_titles_match_values_count(self):
        """get_titles() and get_values() must return the same number of items."""
        sampler = GPUPowerSampler()
        titles = sampler.get_titles()
        values = sampler.get_values()
        assert len(titles) == len(values)
        sampler.close()

    def test_titles_are_tuples(self):
        """Interface requires tuples."""
        sampler = GPUPowerSampler()
        assert isinstance(sampler.get_titles(), tuple)
        assert isinstance(sampler.get_values(), tuple)
        sampler.close()

    def test_close_is_idempotent(self):
        """close() can be called multiple times without error."""
        sampler = GPUPowerSampler()
        sampler.close()
        sampler.close()  # should not raise

    def test_no_gpu_produces_empty_columns(self):
        """On a machine without GPUs, sampler produces zero-length tuples."""
        with patch.object(GPUPowerSampler, '_detect'):
            sampler = GPUPowerSampler.__new__(GPUPowerSampler)
            sampler._platform = None
            sampler._handles = []
            sampler._cleanup = None
            sampler._num_gpus = 0
            assert sampler.get_titles() == ()
            assert sampler.get_values() == ()


class TestGPUSamplerNvidia:
    """Test NVIDIA detection and reading paths."""

    @patch("subprocess.run")
    def test_nvidia_smi_multi_gpu(self, mock_run):
        """Detects multiple GPUs via nvidia-smi."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="245.3\n238.7\n"
        )
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = None
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 0
        result = sampler._try_nvidia_smi()
        assert result is True
        assert sampler._num_gpus == 2
        assert sampler._platform == "nvidia-smi"
        assert sampler.get_titles() == ("GPU0_Power_W", "GPU1_Power_W")

    @patch("subprocess.run")
    def test_nvidia_smi_read(self, mock_run):
        """Reading returns correct wattage values."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="300.5\n295.2\n"
        )
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = "nvidia-smi"
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 2
        values = sampler._read_nvidia_smi()
        assert values == (300.5, 295.2)

    @patch("subprocess.run")
    def test_nvidia_smi_failure(self, mock_run):
        """Returns False when nvidia-smi is not available."""
        mock_run.side_effect = FileNotFoundError
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = None
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 0
        assert sampler._try_nvidia_smi() is False


class TestGPUSamplerRocm:
    """Test ROCm detection paths."""

    ROCM_JSON = '{"card0": {"Average Graphics Package Power (W)": "470.1 W"}, "card1": {"Average Graphics Package Power (W)": "465.3 W"}}'

    @patch("subprocess.run")
    def test_rocm_smi_multi_gpu(self, mock_run):
        """Detects multiple GPUs via rocm-smi JSON."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self.ROCM_JSON
        )
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = None
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 0
        result = sampler._try_rocm_smi()
        assert result is True
        assert sampler._num_gpus == 2

    def test_parse_rocm_smi_multi(self):
        """Multi-GPU JSON parsing extracts per-card power."""
        import json
        data = json.loads(self.ROCM_JSON)
        powers = GPUPowerSampler._parse_rocm_smi_multi(data)
        assert len(powers) == 2
        assert powers[0] == pytest.approx(470.1)
        assert powers[1] == pytest.approx(465.3)


class TestGPUSamplerGaudi:
    """Test Intel Gaudi detection paths."""

    HLSMI_OUTPUT = """
HL-SMI 1.11.0

GPU 0:
    Power Draw                    : 350.5 W
    Power Limit                   : 600 W

GPU 1:
    Power Draw                    : 345.2 W
    Power Limit                   : 600 W
"""

    @patch("subprocess.run")
    def test_hlsmi_multi_gpu(self, mock_run):
        """Detects multiple Gaudi devices via hl-smi."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self.HLSMI_OUTPUT
        )
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = None
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 0
        result = sampler._try_hlsmi()
        assert result is True
        assert sampler._num_gpus == 2

    @patch("subprocess.run")
    def test_hlsmi_read(self, mock_run):
        """Reading returns per-device wattage."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=self.HLSMI_OUTPUT
        )
        sampler = GPUPowerSampler.__new__(GPUPowerSampler)
        sampler._platform = "hl-smi"
        sampler._handles = []
        sampler._cleanup = None
        sampler._num_gpus = 2
        values = sampler._read_hlsmi()
        assert values == (350.5, 345.2)
