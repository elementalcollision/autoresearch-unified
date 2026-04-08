"""Combined power report merging GPU-only and wall-power measurements.

When both GPU power (from PowerMonitor) and wall power (from WallPowerAdapter)
are available, this module correlates them to produce derived metrics:
  - gpu_power_fraction: what % of wall power is the GPU
  - overhead_watts: wall - gpu (CPU, memory, PSU losses, fans)
  - pue_estimate: wall / gpu (system-level power usage effectiveness)
"""

from __future__ import annotations

from dataclasses import dataclass

from backends.wall_power import WallPowerResult


@dataclass
class CombinedPowerReport:
    """Merged power metrics from GPU sampling and wall-power measurement."""

    # GPU metrics (always available when PowerMonitor works)
    gpu_avg_watts: float = 0.0
    gpu_total_joules: float = 0.0
    gpu_joules_per_token: float = 0.0

    # Wall metrics (only when MLCommons server available)
    wall_avg_watts: float = 0.0
    wall_total_joules: float = 0.0
    wall_joules_per_token: float = 0.0

    # Derived (only when both sources available)
    gpu_power_fraction: float = 0.0
    overhead_watts: float = 0.0
    pue_estimate: float = 0.0

    # Source tracking
    measurement_quality: str = "gpu_only"  # "gpu_only" | "wall_only" | "combined"

    @classmethod
    def from_sources(
        cls,
        gpu_watts: float,
        gpu_joules: float,
        wall_data: WallPowerResult | None,
        training_seconds: float,
        total_tokens: int,
    ) -> CombinedPowerReport:
        """Build a combined report from GPU and optional wall-power data."""
        gpu_jpt = gpu_joules / total_tokens if total_tokens > 0 else 0.0
        has_gpu = gpu_watts > 0
        has_wall = wall_data is not None and wall_data.avg_watts > 0

        report = cls(
            gpu_avg_watts=gpu_watts,
            gpu_total_joules=gpu_joules,
            gpu_joules_per_token=gpu_jpt,
        )

        if has_wall:
            wall_joules = wall_data.avg_watts * training_seconds
            wall_jpt = wall_joules / total_tokens if total_tokens > 0 else 0.0
            report.wall_avg_watts = wall_data.avg_watts
            report.wall_total_joules = wall_joules
            report.wall_joules_per_token = wall_jpt

        if has_gpu and has_wall:
            report.gpu_power_fraction = gpu_watts / wall_data.avg_watts
            report.overhead_watts = wall_data.avg_watts - gpu_watts
            report.pue_estimate = wall_data.avg_watts / gpu_watts
            report.measurement_quality = "combined"
        elif has_wall:
            report.measurement_quality = "wall_only"
        elif has_gpu:
            report.measurement_quality = "gpu_only"
        else:
            report.measurement_quality = "gpu_only"

        return report
