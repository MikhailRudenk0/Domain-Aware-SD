#!/usr/bin/env python3
"""
Download models and/or synthetic dataset from S3 (Timeweb Cloud).

Usage:
    python src/download_from_s3.py --all
    python src/download_from_s3.py --models
    python src/download_from_s3.py --data
    python src/download_from_s3.py --models --target-only
    python src/download_from_s3.py --data --version v2
    python src/download_from_s3.py --all --force   # re-download even if file exists
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

BUCKET      = os.getenv("S3_BUCKET",   "domain-aware-sd")
ENDPOINT    = os.getenv("S3_ENDPOINT", "https://s3.twcstorage.ru")
REGION      = os.getenv("S3_REGION",   "ru-1-hot")
ACCESS_KEY  = os.getenv("S3_ACCESS_KEY")
SECRET_KEY  = os.getenv("S3_SECRET_KEY")

# S3 prefix → local directory
MODELS = {
    "target":  ("models/TurboSparse-Mistral-Instruct", PROJECT_ROOT / "TurboSparse-Mistral-Instruct"),
    "drafter": ("models/tiny-mixtral",                 PROJECT_ROOT / "tiny-mixtral"),
}


def make_client():
    if not ACCESS_KEY or not SECRET_KEY:
        sys.exit("ERROR: S3_ACCESS_KEY / S3_SECRET_KEY not set in .env")
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name=REGION,
    )


def list_prefix(client, prefix: str) -> list[dict]:
    """Return all objects under prefix (handles pagination)."""
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def download_object(client, key: str, dest: Path, force: bool) -> tuple[str, str]:
    """
    Download one S3 object to dest.
    Returns (key, status) where status is 'downloaded', 'skipped', or 'error: ...'.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return key, "skipped"
    try:
        client.download_file(BUCKET, key, str(dest))
        return key, "downloaded"
    except ClientError as e:
        return key, f"error: {e}"


def download_prefix(
    client,
    s3_prefix: str,
    local_dir: Path,
    force: bool,
    workers: int = 8,
    label: str = "",
) -> None:
    objects = list_prefix(client, s3_prefix)
    if not objects:
        print(f"  No objects found under s3://{BUCKET}/{s3_prefix}")
        return

    # Build (key, local_path) pairs
    tasks = []
    for obj in objects:
        key = obj["Key"]
        relative = key[len(s3_prefix):].lstrip("/")
        dest = local_dir / relative
        tasks.append((key, dest))

    total_mb = sum(o["Size"] for o in objects) / (1024 ** 2)
    desc = label or s3_prefix
    print(f"  {len(tasks)} files  ({total_mb:.0f} MB)  → {local_dir}")

    downloaded = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_object, client, key, dest, force): key
            for key, dest in tasks
        }
        with tqdm(total=len(futures), unit="file", desc=f"  {desc}") as bar:
            for future in as_completed(futures):
                _, status = future.result()
                if status == "downloaded":
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    tqdm.write(f"    WARN {status}")
                bar.update(1)

    print(f"  done — {downloaded} downloaded, {skipped} skipped, {errors} errors")


def main():
    parser = argparse.ArgumentParser(description="Download assets from S3")
    parser.add_argument("--all",         action="store_true", help="Download models + data")
    parser.add_argument("--models",      action="store_true", help="Download both models")
    parser.add_argument("--data",        action="store_true", help="Download synthetic data")
    parser.add_argument("--target-only", action="store_true", help="Target model only (with --models)")
    parser.add_argument("--version",     default="v1",        help="Synthetic data version (default: v1)")
    parser.add_argument("--force",       action="store_true", help="Re-download even if file exists")
    parser.add_argument("--workers",     type=int, default=8, help="Parallel download threads (default: 8)")
    args = parser.parse_args()

    if not any([args.all, args.models, args.data]):
        parser.print_help()
        sys.exit(1)

    client = make_client()

    # ── Models ────────────────────────────────────────────────────────────────
    if args.all or args.models:
        print("\n── Models ──────────────────────────────────────────")
        keys = ["target"] if args.target_only else list(MODELS.keys())
        for key in keys:
            prefix, local_dir = MODELS[key]
            print(f"\n{key}: s3://{BUCKET}/{prefix}")
            download_prefix(client, prefix, local_dir,
                            force=args.force, workers=args.workers, label=key)

    # ── Synthetic data ────────────────────────────────────────────────────────
    if args.all or args.data:
        print("\n── Synthetic data ──────────────────────────────────")
        prefix    = f"data/synthetic/{args.version}"
        local_dir = PROJECT_ROOT / "data" / "synthetic" / args.version
        print(f"\ns3://{BUCKET}/{prefix}")
        download_prefix(client, prefix, local_dir,
                        force=args.force, workers=args.workers, label=f"data/{args.version}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
