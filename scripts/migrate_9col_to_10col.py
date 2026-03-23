#!/usr/bin/env python3
"""Migrate legacy 9-column results.tsv files to 10-column format.

The Metal/Apple Silicon repo used 9 columns (no gpu_name).
The unified format requires 10 columns. This script adds an empty
gpu_name column to any 9-column TSV files it finds.

Usage:
    python scripts/migrate_9col_to_10col.py              # scan results/ dir
    python scripts/migrate_9col_to_10col.py path/to/results.tsv  # specific file
"""

import os
import sys
from pathlib import Path


HEADER_9 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\n"
HEADER_10 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\n"


def migrate_file(path: Path) -> bool:
    """Migrate a 9-column TSV to 10 columns. Returns True if migrated."""
    with open(path) as f:
        content = f.read()

    if not content.strip():
        return False

    lines = content.split("\n")
    header = lines[0]
    fields = header.strip().split("\t")

    # Already 10 columns
    if len(fields) >= 10 or "gpu_name" in header:
        return False

    # Verify it's 9-column format
    if len(fields) != 9:
        print(f"  SKIP {path}: unexpected {len(fields)} columns")
        return False

    # Migrate: add gpu_name column
    new_lines = [HEADER_10.rstrip()]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 9:
            new_lines.append(line + "\t")  # empty gpu_name
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

    if not paths:
        print("No results.tsv files found.")
        return

    migrated = 0
    for path in paths:
        if migrate_file(path):
            migrated += 1

    print(f"\nDone: {migrated}/{len(paths)} files migrated to 10-column format.")


if __name__ == "__main__":
    main()
