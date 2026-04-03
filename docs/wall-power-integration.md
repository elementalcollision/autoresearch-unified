# Wall-Power Integration with MLCommons power-dev

## Overview

autoresearch-unified supports optional wall-power measurement using the
[MLCommons power-dev](https://github.com/mlcommons/power-dev) framework.
When enabled, this provides total system AC power from a calibrated Yokogawa
power meter alongside the existing GPU-only software power readings.

## Architecture

```
Training Script
  ├── PowerMonitor (GPU-only, always active)     → avg_watts, total_joules
  ├── WallPowerAdapter (opt-in, env var gated)    → wall_watts from Yokogawa
  └── CombinedPowerReport (merges both streams)   → derived metrics
```

The existing `PowerMonitor` is **unchanged**. The `WallPowerAdapter` runs as
a sidecar that communicates with a running MLCommons power-dev server.

## Setup

### Prerequisites

1. A Yokogawa power meter (WT210, WT310, or WT330E) connected to:
   - The AC power line of the System Under Test
   - A director machine via RS232, GPIB, TCP, or USB
2. The MLCommons power-dev framework installed on the director machine
3. Network connectivity between the SUT and the director

### Install MLCommons power-dev

```bash
git clone https://github.com/mlcommons/power-dev.git
cd power-dev
pip install -e .
```

### Configure the power server

Copy and edit the server configuration template:

```bash
cp ptd_client_server/server.template.conf server.conf
```

Edit `server.conf` to match your Yokogawa meter connection:

```ini
[server]
ntpServer = time.google.com

[ptd]
ptd = /path/to/PTDaemon
deviceType = 49          # 49=WT310, 52=WT330E, 8=WT210
interfaceFlag = 4        # TCP
devicePort = 192.168.1.100  # Yokogawa IP
```

### Start the power server

```bash
power_server -c server.conf -p 4950 -o /tmp/power_logs
```

### Enable wall-power measurement in autoresearch

Set environment variables before running training:

```bash
export AUTORESEARCH_WALL_POWER=1
export AUTORESEARCH_WALL_POWER_HOST=192.168.1.50   # director machine IP
export AUTORESEARCH_WALL_POWER_PORT=4950            # default
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTORESEARCH_WALL_POWER` | (unset) | Set to `1` to enable wall-power measurement |
| `AUTORESEARCH_WALL_POWER_HOST` | `localhost` | MLCommons power-dev server hostname/IP |
| `AUTORESEARCH_WALL_POWER_PORT` | `4950` | Server TCP port |

## Output

### New metrics in final summary

When wall-power is available, training scripts print additional lines:

```
wall_watts:            850.3
wall_joules_per_token: 0.000425
wall_total_energy_j:   255090.0
gpu_power_fraction:    0.5528
```

### New TSV columns

Four columns are appended to results.tsv (columns 15-18):

| Column | Type | Description |
|--------|------|-------------|
| `wall_watts` | float | Average wall power (W). 0.0 if unavailable |
| `wall_joules_per_token` | float | Wall energy per token (J/tok) |
| `wall_total_energy_joules` | float | Total wall energy (J) |
| `gpu_power_fraction` | float | gpu_watts / wall_watts |

### Derived metrics

When both GPU and wall power are available:

- **GPU Power Fraction**: What percentage of wall power is consumed by the GPU
  (typically 60-85% for dedicated GPU servers)
- **Overhead Watts**: `wall_watts - gpu_watts` — CPU, memory, PSU losses, fans
- **PUE Estimate**: `wall_watts / gpu_watts` — system-level power usage
  effectiveness approximation

## Graceful Degradation

Wall-power measurement follows the same philosophy as GPU power monitoring:

- If `AUTORESEARCH_WALL_POWER` is not set to `1`, all wall-power methods are no-ops
- If the server is unreachable, `start()` logs a warning and returns
- If the connection drops mid-run, `stop()` returns zeros
- Training **never** crashes due to wall-power measurement

## Migration

To migrate existing 14-column results.tsv files to the new 18-column format:

```bash
python scripts/migrate_14col_to_18col.py
```

Old files get `0.0` for all wall-power columns. The `load_results()` function
handles both 14-column and 18-column files automatically via `dict.get()`.

## MLCommons GPU Sampler (upstream contribution)

`backends/mlcommons_sampler.py` is a standalone sampler that conforms to the
MLCommons `power_meter_sampling` plugin interface. It provides per-GPU software
power readings that complement their Yokogawa wall-power measurements.

To use it with MLCommons power-dev, copy `mlcommons_sampler.py` into the
`power_meter_sampling/samplers/` directory in the power-dev repository.

```python
from mlcommons_sampler import GPUPowerSampler

sampler = GPUPowerSampler()
print(sampler.get_titles())   # ("GPU0_Power_W", "GPU1_Power_W", ...)
print(sampler.get_values())   # (245.3, 238.7, ...)
sampler.close()
```
