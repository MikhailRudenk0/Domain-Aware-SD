#!/usr/bin/env python3
"""
Upload to / download from S3 (Timeweb Cloud).

Download:
    python src/download_from_s3.py --all                  # models + synthetic data
    python src/download_from_s3.py --models
    python src/download_from_s3.py --models --target-only
    python src/download_from_s3.py --data
    python src/download_from_s3.py --flan                 # raw Flan dataset
    python src/download_from_s3.py --data --version v2
    python src/download_from_s3.py --all --force          # re-download even if exists

Upload:
    python src/download_from_s3.py --upload-flan          # upload flan/ → S3
    python src/download_from_s3.py --upload-flan --force  # re-upload even if key exists

Common:
    --workers N   parallel threads (default: 8)
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

BUCKET     = os.getenv("S3_BUCKET",   "domain-aware-sd")
ENDPOINT   = os.getenv("S3_ENDPOINT", "https://s3.twcstorage.ru")
REGION     = os.getenv("S3_REGION",   "ru-1-hot")
ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
SECRET_KEY = os.getenv("S3_SECRET_KEY")

MODELS = {
    "target":  ("models/TurboSparse-Mistral-Instruct", PROJECT_ROOT / "TurboSparse-Mistral-Instruct"),
    "drafter": ("models/tiny-mixtral",                 PROJECT_ROOT / "tiny-mixtral"),
}

FLAN_S3_PREFIX = "flan"
FLAN_LOCAL_DIR = PROJECT_ROOT / "flan"


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


# ── S3 helpers ────────────────────────────────────────────────────────────────

def list_prefix(client, prefix: str) -> list[dict]:
    objects = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        objects.extend(page.get("Contents", []))
    return objects


def key_exists(client, key: str) -> bool:
    try:
        client.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


# ── Download ──────────────────────────────────────────────────────────────────

def _download_one(client, key: str, dest: Path, force: bool) -> tuple[str, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return key, "skipped"
    try:
        client.download_file(BUCKET, key, str(dest))
        return key, "downloaded"
    except ClientError as e:
        return key, f"error: {e}"


def download_prefix(client, s3_prefix: str, local_dir: Path,
                    force: bool, workers: int, label: str = "") -> None:
    objects = list_prefix(client, s3_prefix)
    if not objects:
        print(f"  No objects found under s3://{BUCKET}/{s3_prefix}")
        return

    tasks = [
        (obj["Key"], local_dir / obj["Key"][len(s3_prefix):].lstrip("/"))
        for obj in objects
    ]
    total_mb = sum(o["Size"] for o in objects) / (1024 ** 2)
    print(f"  {len(tasks)} files  ({total_mb:.0f} MB)  → {local_dir}")

    downloaded = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_one, client, key, dest, force): key
            for key, dest in tasks
        }
        with tqdm(total=len(futures), unit="file", desc=f"  {label or s3_prefix}") as bar:
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


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload_one(client, local: Path, key: str, force: bool) -> tuple[str, str]:
    if not force and key_exists(client, key):
        return key, "skipped"
    try:
        client.upload_file(str(local), BUCKET, key)
        return key, "uploaded"
    except ClientError as e:
        return key, f"error: {e}"


def upload_dir(client, local_dir: Path, s3_prefix: str,
               force: bool, workers: int, label: str = "") -> None:
    if not local_dir.exists():
        sys.exit(f"ERROR: local directory not found: {local_dir}")

    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    if not files:
        print(f"  No files found in {local_dir}")
        return

    tasks = [
        (f, f"{s3_prefix}/{f.relative_to(local_dir)}")
        for f in files
    ]
    total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
    print(f"  {len(tasks)} files  ({total_mb:.0f} MB)  → s3://{BUCKET}/{s3_prefix}")

    uploaded = skipped = errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_upload_one, client, local, key, force): key
            for local, key in tasks
        }
        with tqdm(total=len(futures), unit="file", desc=f"  {label or s3_prefix}") as bar:
            for future in as_completed(futures):
                _, status = future.result()
                if status == "uploaded":
                    uploaded += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
                    tqdm.write(f"    WARN {status}")
                bar.update(1)

    print(f"  done — {uploaded} uploaded, {skipped} skipped, {errors} errors")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Upload/download project assets to/from S3")

    # Download flags
    dl = parser.add_argument_group("download")
    dl.add_argument("--all",         action="store_true", help="Download models + synthetic data")
    dl.add_argument("--models",      action="store_true", help="Download both models")
    dl.add_argument("--data",        action="store_true", help="Download synthetic data")
    dl.add_argument("--flan",        action="store_true", help="Download raw Flan dataset from S3")
    dl.add_argument("--target-only", action="store_true", help="Target model only (use with --models)")
    dl.add_argument("--version",     default="v1",        help="Synthetic data version (default: v1)")

    # Upload flags
    ul = parser.add_argument_group("upload")
    ul.add_argument("--upload-flan", action="store_true", help="Upload flan/ directory to S3")

    # Common
    parser.add_argument("--force",   action="store_true", help="Re-transfer even if file already exists")
    parser.add_argument("--workers", type=int, default=8, help="Parallel threads (default: 8)")

    args = parser.parse_args()

    any_action = any([args.all, args.models, args.data, args.flan, args.upload_flan])
    if not any_action:
        parser.print_help()
        sys.exit(1)

    client = make_client()

    # ── Upload: flan ──────────────────────────────────────────────────────────
    if args.upload_flan:
        print(f"\n── Upload flan → s3://{BUCKET}/{FLAN_S3_PREFIX} ──────────────────")
        upload_dir(client, FLAN_LOCAL_DIR, FLAN_S3_PREFIX,
                   force=args.force, workers=args.workers, label="flan")

    # ── Download: models ──────────────────────────────────────────────────────
    if args.all or args.models:
        print("\n── Models ──────────────────────────────────────────")
        keys = ["target"] if args.target_only else list(MODELS.keys())
        for key in keys:
            prefix, local_dir = MODELS[key]
            print(f"\n{key}: s3://{BUCKET}/{prefix}")
            download_prefix(client, prefix, local_dir,
                            force=args.force, workers=args.workers, label=key)

    # ── Download: synthetic data ──────────────────────────────────────────────
    if args.all or args.data:
        print("\n── Synthetic data ──────────────────────────────────")
        prefix    = f"data/synthetic/{args.version}"
        local_dir = PROJECT_ROOT / "data" / "synthetic" / args.version
        print(f"\ns3://{BUCKET}/{prefix}")
        download_prefix(client, prefix, local_dir,
                        force=args.force, workers=args.workers, label=f"data/{args.version}")

    # ── Download: flan ────────────────────────────────────────────────────────
    if args.flan:
        print(f"\n── Flan dataset ─────────────────────────────────────")
        print(f"\ns3://{BUCKET}/{FLAN_S3_PREFIX}")
        download_prefix(client, FLAN_S3_PREFIX, FLAN_LOCAL_DIR,
                        force=args.force, workers=args.workers, label="flan")

    print("\nAll done.")


if __name__ == "__main__":
    main()
