#!/usr/bin/env python3
"""Migrate 10-column results.tsv files to 11-column format.

The unified repo originally used 10 columns (no baseline_sha).
The current format requires 11 columns. This script adds an empty
baseline_sha column to any 10-column TSV files it finds.

Usage:
    python scripts/migrate_10col_to_11col.py              # scan results/ dir
    python scripts/migrate_10col_to_11col.py path/to/results.tsv  # specific file
"""

import os
import sys
from pathlib import Path


HEADER_10 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name"
HEADER_11 = "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps\tstatus\tnotes\tgpu_name\tbaseline_sha"


def migrate_file(path: Path) -> bool:
    """Migrate a 10-column TSV to 11 columns. Returns True if migrated."""
    with open(path) as f:
        content = f.read()

    if not content.strip():
        return False

    lines = content.split("\n")
    header = lines[0].strip()
    fields = header.split("\t")

    # Already 11 columns
    if len(fields) >= 11 or "baseline_sha" in header:
        return False

    # Verify it's 10-column format
    if len(fields) != 10:
        print(f"  SKIP {path}: unexpected {len(fields)} columns")
        return False

    # Migrate: add baseline_sha column
    new_lines = [HEADER_11]
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) == 10:
            new_lines.append(line + "\t")  # empty baseline_sha
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

    print(f"\nDone: {migrated}/{len(paths)} files migrated to 11-column format.")


if __name__ == "__main__":
    main()
