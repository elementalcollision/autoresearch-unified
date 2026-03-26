#!/usr/bin/env python3
"""Migrate 11-column results.tsv files to 14-column format.

Adds three energy instrumentation columns: watts, joules_per_token,
total_energy_joules. Existing experiments get 0.0 for all three
(energy data was not collected for them).

Usage:
    python scripts/migrate_11col_to_14col.py              # scan results/ dir
    python scripts/migrate_11col_to_14col.py path/to/results.tsv  # specific file
"""

import os
import sys
from pathlib import Path


HEADER_11 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha"
HEADER_14 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha\twatts\tjoules_per_token\ttotal_energy_joules"


def migrate_file(path: Path) -> bool:
    """Migrate an 11-column TSV to 14 columns. Returns True if migrated."""
    with open(path) as f:
        content = f.read()

    if not content.strip():
        return False

    lines = content.split("\n")
    header = lines[0].strip()
    fields = header.split("\t")

    # Already 14 columns
    if len(fields) >= 14 or "total_energy_joules" in header:
        return False

    # Verify it's 11-column format
    if len(fields) != 11:
        print(f"  SKIP {path}: unexpected {len(fields)} columns (expected 11)")
        return False

    # Migrate: add watts, joules_per_token, total_energy_joules columns
    new_lines = [HEADER_14]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 11:
            new_lines.append(line + "\t0.0\t0.000000\t0.0")
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

    print(f"\nDone: {migrated}/{len(paths)} files migrated to 14-column format.")


if __name__ == "__main__":
    main()
