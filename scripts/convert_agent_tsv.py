#!/usr/bin/env python3
"""Convert agent_results_*.tsv (6-column, our fork) to the 14-column format
used by autoresearch-unified.

Our fork format (6 columns):
    commit  val_bpb  memory_gb  status  description  sample_text

autoresearch-unified format (14 columns):
    exp  description  val_bpb  peak_mem_gb  tok_sec  mfu  steps  status
    notes  gpu_name  baseline_sha  watts  joules_per_token  total_energy_joules

Usage:
    python scripts/convert_agent_tsv.py INPUT.tsv OUTPUT.tsv --tag kimi-k25-r1 --gpu "NVIDIA GeForce RTX 5070 Ti"
"""

import argparse
import csv
import sys


def convert(input_path, output_path, tag, gpu_name, baseline_sha_override=None):
    rows = []
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"ERROR: No data rows in {input_path}", file=sys.stderr)
        sys.exit(1)

    # Detect baseline SHA from first row (or override)
    first_sha = rows[0].get("commit", "").strip()
    baseline_sha = baseline_sha_override or first_sha

    with open(output_path, "w", newline="") as f:
        f.write(
            "exp\tdescription\tval_bpb\tpeak_mem_gb\ttok_sec\tmfu\tsteps"
            "\tstatus\tnotes\tgpu_name\tbaseline_sha"
            "\twatts\tjoules_per_token\ttotal_energy_joules\n"
        )

        for i, row in enumerate(rows):
            commit = (row.get("commit") or "").strip()
            val_bpb = (row.get("val_bpb") or "0").strip()
            memory_gb = (row.get("memory_gb") or "0").strip()
            status_raw = (row.get("status") or "").strip().lower()
            description = (row.get("description") or "").strip()

            # Map status: first row with status "keep" and description
            # "baseline" should become "baseline"
            if i == 0 and "baseline" in description.lower():
                status = "baseline"
            elif status_raw == "keep":
                status = "keep"
            elif status_raw == "fail" or status_raw == "crash":
                status = "crash"
            elif status_raw == "skip" or status_raw == "":
                status = "skip"
            else:
                status = "discard"

            # For crashes, val_bpb should be 0
            if status in ("crash", "skip"):
                val_bpb = "0.000000"

            exp = f"exp{i}"
            notes = f"{tag}-controlled"

            f.write(
                f"{exp}\t{description}\t{val_bpb}\t{memory_gb}\t0\t0.0\t0"
                f"\t{status}\t{notes}\t{gpu_name}\t{baseline_sha}"
                f"\t0.0\t0.000000\t0.0\n"
            )

    print(f"Converted {len(rows)} rows: {input_path} -> {output_path}")

    # Summary
    statuses = {}
    for i, row in enumerate(rows):
        s = (row.get("status") or "").strip().lower()
        if i == 0 and "baseline" in (row.get("description") or "").lower():
            s = "baseline"
        statuses[s] = statuses.get(s, 0) + 1
    print(f"  Status breakdown: {statuses}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 6-column agent TSV to 14-column unified format"
    )
    parser.add_argument("input", help="Input TSV (agent_results_*.tsv)")
    parser.add_argument("output", help="Output TSV (results_*_r1.tsv)")
    parser.add_argument("--tag", required=True, help="Run tag (e.g. kimi-k25-r1)")
    parser.add_argument(
        "--gpu", default="NVIDIA GeForce RTX 5070 Ti",
        help="GPU name for the gpu_name column"
    )
    parser.add_argument(
        "--baseline-sha", default=None,
        help="Override baseline commit SHA (default: first row's commit)"
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.tag, args.gpu, args.baseline_sha)


if __name__ == "__main__":
    main()
