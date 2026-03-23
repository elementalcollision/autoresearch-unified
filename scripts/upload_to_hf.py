#!/usr/bin/env python3
"""
Upload consolidated Parquet files and metadata to HuggingFace.

Usage:
  python scripts/upload_to_hf.py                    # Upload all files
  python scripts/upload_to_hf.py --dry-run           # Show what would be uploaded
  python scripts/upload_to_hf.py --repo-id USER/NAME # Override repo ID

Prerequisites:
  pip install huggingface_hub
  huggingface-cli login
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DEFAULT_REPO_ID = "davegraham/autoresearch-experiments"
BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Upload to HuggingFace")
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HuggingFace dataset repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without uploading",
    )
    args = parser.parse_args()

    # Files to upload: (local_path, path_in_repo)
    uploads = [
        (BASE_DIR / "data" / "experiments.parquet", "data/experiments.parquet"),
        (BASE_DIR / "data" / "hardware.parquet", "data/hardware.parquet"),
        (BASE_DIR / "huggingface" / "README.md", "README.md"),
        (BASE_DIR / "huggingface" / "croissant.json", "croissant.json"),
    ]

    # Validate all files exist
    for local_path, _ in uploads:
        if not local_path.exists():
            print(f"ERROR: {local_path} not found. Run consolidate_results.py first.")
            return

    if args.dry_run:
        print(f"Would upload to: {args.repo_id}")
        for local_path, repo_path in uploads:
            size_kb = local_path.stat().st_size / 1024
            print(f"  {local_path.name} -> {repo_path} ({size_kb:.1f} KB)")
        return

    api = HfApi()

    # Create repo if it doesn't exist
    print(f"Creating/verifying repo: {args.repo_id}")
    create_repo(args.repo_id, repo_type="dataset", exist_ok=True)

    # Upload files
    for local_path, repo_path in uploads:
        print(f"Uploading {local_path.name} -> {repo_path}")
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
