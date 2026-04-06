"""Hardware drift detection — thermal throttling, memory pressure, power anomalies."""

from dataclasses import dataclass

from autoresearch.results import ExperimentResult


@dataclass
class HardwareAlert:
    """A single hardware drift alert."""
    level: str       # "info", "warning", "critical"
    category: str    # "thermal", "memory", "power"
    message: str


@dataclass
class HardwareConfig:
    """Tunable thresholds for hardware drift detection."""
    memory_trend_window: int = 10     # experiments to check for memory trend
    memory_growth_pct: float = 15.0   # peak_mem growth > this % = pressure
    power_efficiency_window: int = 10
    power_efficiency_drop_pct: float = 20.0  # joules_per_token increase > this %
    thermal_toksec_drop_pct: float = 15.0    # tok/sec drop + watts drop = throttling


class HardwareDrift:
    """Detects hardware-related drift from experiment results."""

    def __init__(self, config: HardwareConfig | None = None):
        self.config = config or HardwareConfig()

    def check(self, results: list[ExperimentResult]) -> list[HardwareAlert]:
        alerts = []
        alerts.extend(self._check_memory_pressure(results))
        alerts.extend(self._check_power_efficiency(results))
        alerts.extend(self._check_thermal_throttling(results))
        return alerts

    def _check_memory_pressure(self, results: list[ExperimentResult]) -> list[HardwareAlert]:
        """Detect if peak memory is trending upward (possible leak)."""
        recent = [r for r in results if r.peak_mem_gb > 0]
        if len(recent) < self.config.memory_trend_window:
            return []

        window = recent[-self.config.memory_trend_window:]
        first_half = window[:len(window) // 2]
        second_half = window[len(window) // 2:]

        avg_first = sum(r.peak_mem_gb for r in first_half) / len(first_half)
        avg_second = sum(r.peak_mem_gb for r in second_half) / len(second_half)

        if avg_first > 0:
            growth_pct = (avg_second - avg_first) / avg_first * 100
            if growth_pct > self.config.memory_growth_pct:
                return [HardwareAlert(
                    level="warning",
                    category="memory",
                    message=f"Memory pressure: peak memory grew {growth_pct:.1f}% over "
                            f"last {self.config.memory_trend_window} experiments "
                            f"({avg_first:.1f}GB -> {avg_second:.1f}GB). "
                            f"Possible memory leak or cumulative fragmentation.",
                )]
        return []

    def _check_power_efficiency(self, results: list[ExperimentResult]) -> list[HardwareAlert]:
        """Detect if energy per token is increasing without val_bpb improvement."""
        recent = [r for r in results if r.joules_per_token > 0]
        if len(recent) < self.config.power_efficiency_window:
            return []

        window = recent[-self.config.power_efficiency_window:]
        first_half = window[:len(window) // 2]
        second_half = window[len(window) // 2:]

        avg_first_jpt = sum(r.joules_per_token for r in first_half) / len(first_half)
        avg_second_jpt = sum(r.joules_per_token for r in second_half) / len(second_half)

        if avg_first_jpt > 0:
            increase_pct = (avg_second_jpt - avg_first_jpt) / avg_first_jpt * 100
            if increase_pct > self.config.power_efficiency_drop_pct:
                # Check if val_bpb improved enough to justify the cost
                first_bpb = [r.val_bpb for r in first_half if r.val_bpb > 0]
                second_bpb = [r.val_bpb for r in second_half if r.val_bpb > 0]

                bpb_improved = False
                if first_bpb and second_bpb:
                    bpb_improved = min(second_bpb) < min(first_bpb)

                if not bpb_improved:
                    return [HardwareAlert(
                        level="warning",
                        category="power",
                        message=f"Energy efficiency degradation: joules_per_token increased "
                                f"{increase_pct:.1f}% without val_bpb improvement. "
                                f"({avg_first_jpt:.4f} -> {avg_second_jpt:.4f} J/tok). "
                                f"Consider whether recent changes increased compute without benefit.",
                    )]
        return []

    def _check_thermal_throttling(self, results: list[ExperimentResult]) -> list[HardwareAlert]:
        """Detect simultaneous watts drop + tok/sec drop (thermal throttling signature)."""
        recent = [r for r in results if r.watts > 0 and r.tok_sec > 0]
        if len(recent) < 5:
            return []

        # Compare last experiment to rolling average of previous 5
        prev = recent[-6:-1]
        latest = recent[-1]

        avg_watts = sum(r.watts for r in prev) / len(prev)
        avg_toksec = sum(r.tok_sec for r in prev) / len(prev)

        if avg_watts > 0 and avg_toksec > 0:
            watts_drop_pct = (avg_watts - latest.watts) / avg_watts * 100
            toksec_drop_pct = (avg_toksec - latest.tok_sec) / avg_toksec * 100

            if watts_drop_pct > 10 and toksec_drop_pct > self.config.thermal_toksec_drop_pct:
                return [HardwareAlert(
                    level="critical",
                    category="thermal",
                    message=f"Possible thermal throttling: watts dropped {watts_drop_pct:.1f}% "
                            f"({avg_watts:.0f}W -> {latest.watts:.0f}W) and tok/sec dropped "
                            f"{toksec_drop_pct:.1f}% ({avg_toksec:.0f} -> {latest.tok_sec}). "
                            f"Check GPU temperature and cooling.",
                )]
        return []
