#!/usr/bin/env python3
"""Migrate 14-column results.tsv files to 18-column format.

Adds four wall-power columns: wall_watts, wall_joules_per_token,
wall_total_energy_joules, gpu_power_fraction. Existing experiments
get 0.0 for all four (wall power was not collected for them).

Usage:
    python scripts/migrate_14col_to_18col.py              # scan results/ dir
    python scripts/migrate_14col_to_18col.py path/to/results.tsv  # specific file
"""

import os
import sys
from pathlib import Path


HEADER_14 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules"
HEADER_18 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules\twall_watts\twall_joules_per_token\twall_total_energy_joules\tgpu_power_fraction"


def migrate_file(path: Path) -> bool:
    """Migrate a 14-column TSV to 18 columns. Returns True if migrated."""
    with open(path) as f:
        content = f.read()

    if not content.strip():
        return False

    lines = content.split("\n")
    header = lines[0].strip()
    fields = header.split("\t")

    # Already 18 columns
    if len(fields) >= 18 or "gpu_power_fraction" in header:
        return False

    # Verify it's 14-column format
    if len(fields) != 14:
        print(f"  SKIP {path}: unexpected {len(fields)} columns (expected 14)")
        return False

    # Migrate: add wall_watts, wall_joules_per_token, wall_total_energy_joules, gpu_power_fraction
    new_lines = [HEADER_18]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 14:
            new_lines.append(line + "\t0.0\t0.000000\t0.0\t0.0000")
        else:
            new_lines.append(line)  # leave as-is

    new_content = "\n".join(new_lines) + "\n"

    # Write atomically
    tmp_path = str(path) + ".tmp"
    with open(tmp_path, "w") as f:
        f.write(new_content)
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp_path, str(path))

    print(f"  MIGRATED {path}: {len(new_lines)-1} rows")
    return True


def main():
    if len(sys.argv) > 1:
        paths = [Path(sys.argv[1])]
    else:
        results_dir = Path(__file__).parent.parent / "results"
        paths = list(results_dir.rglob("results.tsv"))
        # Also check root-level results files
        root = Path(__file__).parent.parent
        paths.extend(root.glob("results_*.tsv"))

    if not paths:
        print("No results.tsv files found.")
        return

    migrated = 0
    for path in paths:
        if migrate_file(path):
            migrated += 1

    print(f"\nDone: {migrated}/{len(paths)} files migrated to 18-column format.")


if __name__ == "__main__":
    main()
