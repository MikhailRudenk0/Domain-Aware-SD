#!/usr/bin/env python3
"""
Tokenizer compatibility check for speculative decoding.

Checks which draft model candidates share the same (or subset) vocabulary
as the target model (TurboSparse-Mistral-Instruct), then uploads compatible
models to S3.

Compatibility criteria:
1. Same base tokenizer family (SentencePiece / LlamaTokenizer, not T5/BPE)
2. Draft vocab is a subset of target vocab: every token ID in [0, draft_vocab_size)
   maps to the same string in both tokenizers.

Usage:
    python src/check_tokenizers.py
    python src/check_tokenizers.py --upload      # also upload compatible models to S3
    python src/check_tokenizers.py --upload --dry-run
"""

import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

PROJECT_ROOT = Path(__file__).parent.parent

TARGET_MODEL = PROJECT_ROOT / "TurboSparse-Mistral-Instruct"
DRAFT_CANDIDATES = [
    PROJECT_ROOT / "tiny-mixtral",
    PROJECT_ROOT / "flant5-tuned-30",
    PROJECT_ROOT / "t5-small-finetuned",
]

S3_BUCKET = os.getenv("S3_BUCKET", "domain-aware-sd")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://s3.twcstorage.ru")
S3_REGION = os.getenv("S3_REGION", "ru-1-hot")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")


# ---------------------------------------------------------------------------
# Tokenizer loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_path: Path):
    """Load tokenizer, returning (tokenizer, class_name) or (None, error_msg)."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True,
        )
        return tok, type(tok).__name__
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Compatibility checks
# ---------------------------------------------------------------------------

def get_tokenizer_family_from_config(model_path: Path) -> str:
    """
    Read tokenizer_class from tokenizer_config.json — more reliable than
    inspecting the loaded fast-tokenizer backend (which serializes as BPE).
    """
    cfg_path = model_path / "tokenizer_config.json"
    if not cfg_path.exists():
        return "unknown"
    cfg = json.loads(cfg_path.read_text())
    cls = cfg.get("tokenizer_class", "").lower()
    if "t5" in cls:
        return "t5"
    if "llama" in cls or "mistral" in cls:
        return "sentencepiece"
    if "gpt2" in cls or "roberta" in cls:
        return "bpe"
    return "unknown"


def sp_model_hash(model_path: Path) -> Optional[str]:
    """SHA-256 of the SentencePiece model file, if present."""
    for fname in ("tokenizer.model", "spiece.model"):
        p = model_path / fname
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    return None


def compare_vocab(target_tok, draft_tok, sample_size: int = 2000) -> tuple[bool, str]:
    """
    Check that draft vocab is a subset of target vocab.

    Samples `sample_size` token IDs from [0, min(draft_vocab, target_vocab))
    and verifies they decode to the same string in both tokenizers.
    Returns (is_compatible, explanation).
    """
    draft_vocab = draft_tok.vocab_size
    target_vocab = target_tok.vocab_size

    if draft_vocab > target_vocab:
        return False, (
            f"Draft vocab ({draft_vocab}) is LARGER than target ({target_vocab}). "
            "Draft must be a subset."
        )

    # Sample token IDs to check (skip known special tokens at boundaries)
    check_up_to = min(draft_vocab, target_vocab)
    ids_to_check = list(range(3, min(check_up_to, 100)))  # first 97 regular tokens
    # Also sample from the middle and end of shared range
    import random
    random.seed(42)
    ids_to_check += random.sample(range(100, check_up_to), min(sample_size - 97, check_up_to - 100))

    mismatches = []
    for tok_id in ids_to_check:
        try:
            draft_str = draft_tok.convert_ids_to_tokens(tok_id)
            target_str = target_tok.convert_ids_to_tokens(tok_id)
            if draft_str != target_str:
                mismatches.append((tok_id, draft_str, target_str))
                if len(mismatches) >= 5:
                    break
        except Exception:
            pass

    if mismatches:
        examples = "; ".join(
            f"id={m[0]}: draft={repr(m[1])} vs target={repr(m[2])}"
            for m in mismatches[:3]
        )
        return False, f"Vocab mismatch at {len(mismatches)} checked positions. Examples: {examples}"

    return True, (
        f"Draft vocab ({draft_vocab}) is a subset of target vocab ({target_vocab}). "
        f"Checked {len(ids_to_check)} token IDs — all match."
    )


def check_compatibility(target_path: Path, draft_path: Path) -> dict:
    """Full compatibility report for one draft model."""
    result = {
        "draft": draft_path.name,
        "compatible": False,
        "reason": "",
        "details": {},
    }

    # Load target
    target_tok, target_cls = load_tokenizer(target_path)
    if target_tok is None:
        result["reason"] = f"Could not load target tokenizer: {target_cls}"
        return result

    # Load draft
    draft_tok, draft_cls = load_tokenizer(draft_path)
    if draft_tok is None:
        result["reason"] = f"Could not load draft tokenizer: {draft_cls}"
        return result

    result["details"]["target_class"] = target_cls
    result["details"]["draft_class"] = draft_cls
    result["details"]["target_vocab_size"] = target_tok.vocab_size
    result["details"]["draft_vocab_size"] = draft_tok.vocab_size

    # 1. Family check (read from config, not from loaded tokenizer class)
    target_family = get_tokenizer_family_from_config(target_path)
    draft_family = get_tokenizer_family_from_config(draft_path)
    result["details"]["target_family"] = target_family
    result["details"]["draft_family"] = draft_family

    if target_family != draft_family:
        result["reason"] = (
            f"Tokenizer family mismatch: target={target_family}, draft={draft_family}. "
            "Speculative decoding requires identical tokenization."
        )
        return result

    # 2. SentencePiece model hash (fast path for identical tokenizers)
    target_hash = sp_model_hash(target_path)
    draft_hash = sp_model_hash(draft_path)
    result["details"]["sp_model_hash_match"] = (
        target_hash == draft_hash if target_hash and draft_hash else "n/a"
    )

    # 3. Vocab subset check
    compatible, explanation = compare_vocab(target_tok, draft_tok)
    result["compatible"] = compatible
    result["reason"] = explanation
    result["details"]["vocab_check"] = explanation

    return result


# ---------------------------------------------------------------------------
# S3 upload
# ---------------------------------------------------------------------------

def get_s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        config=Config(
            connect_timeout=120,
            read_timeout=600,
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def upload_model_to_s3(model_path: Path, s3_prefix: str, dry_run: bool = False) -> list[str]:
    """Upload all files in model_path to S3 under s3_prefix. Returns list of uploaded keys."""
    import boto3
    from boto3.s3.transfer import TransferConfig

    SKIP_DIRS = {".git", ".claude", "__pycache__", "runs", "tmp-checkpoint-500"}
    SKIP_SUFFIXES = {".bin"}  # skip pytorch .bin duplicates if safetensors exist
    files = [
        f for f in model_path.rglob("*")
        if f.is_file()
        and not any(part in SKIP_DIRS for part in f.parts)
        and f.suffix not in SKIP_SUFFIXES
    ]
    uploaded = []

    transfer_cfg = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,     # 64 MB: use multipart above this
        multipart_chunksize=64 * 1024 * 1024,     # 64 MB chunks (fewer parts, less reconnect risk)
        max_concurrency=4,
        use_threads=True,
    )

    if not dry_run:
        s3 = get_s3_client()

    for local_file in sorted(files):
        rel = local_file.relative_to(model_path)
        s3_key = f"{s3_prefix}/{rel}".replace("\\", "/")
        size_mb = local_file.stat().st_size / (1024 ** 2)

        if dry_run:
            print(f"  [dry-run] Would upload: {local_file.name} ({size_mb:.1f} MB) → s3://{S3_BUCKET}/{s3_key}")
        else:
            # Check if already uploaded (skip if exists)
            try:
                s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
                print(f"  [skip] {local_file.name} already on S3")
                uploaded.append(s3_key)
                continue
            except Exception:
                pass

            print(f"  Uploading {local_file.name} ({size_mb:.1f} MB) → s3://{S3_BUCKET}/{s3_key} ...")
            s3.upload_file(
                str(local_file),
                S3_BUCKET,
                s3_key,
                Config=transfer_cfg,
            )

        uploaded.append(s3_key)

    return uploaded


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check tokenizer compatibility and optionally upload to S3")
    parser.add_argument("--upload", action="store_true", help="Upload compatible models to S3")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without doing it")
    parser.add_argument("--also-upload-target", action="store_true", help="Also upload the target model to S3")
    args = parser.parse_args()

    print("=" * 70)
    print("Tokenizer Compatibility Check for Speculative Decoding")
    print(f"Target: {TARGET_MODEL.name}")
    print("=" * 70)

    compatible_models = []
    results = []

    for draft_path in DRAFT_CANDIDATES:
        if not draft_path.exists():
            print(f"\n[SKIP] {draft_path.name}: directory not found")
            continue

        print(f"\nChecking: {draft_path.name}")
        report = check_compatibility(TARGET_MODEL, draft_path)
        results.append(report)

        status = "COMPATIBLE" if report["compatible"] else "INCOMPATIBLE"
        print(f"  Status : {status}")
        print(f"  Reason : {report['reason']}")
        if report["details"]:
            d = report["details"]
            print(f"  Target : {d.get('target_class', '?')} | vocab={d.get('target_vocab_size', '?')}")
            print(f"  Draft  : {d.get('draft_class', '?')} | vocab={d.get('draft_vocab_size', '?')}")
            if "sp_model_hash_match" in d:
                print(f"  SP hash match: {d['sp_model_hash_match']}")

        if report["compatible"]:
            compatible_models.append(draft_path)

    print("\n" + "=" * 70)
    print(f"Compatible draft models: {[m.name for m in compatible_models]}")
    print("=" * 70)

    if args.upload or args.dry_run:
        if not compatible_models and not args.also_upload_target:
            print("\nNo compatible models to upload.")
        else:
            print("\nUploading to S3...")
            for model_path in compatible_models:
                s3_prefix = f"models/{model_path.name}"
                print(f"\n  Model: {model_path.name} → s3://{S3_BUCKET}/{s3_prefix}/")
                upload_model_to_s3(model_path, s3_prefix, dry_run=args.dry_run)

            if args.also_upload_target:
                s3_prefix = f"models/{TARGET_MODEL.name}"
                print(f"\n  Target: {TARGET_MODEL.name} → s3://{S3_BUCKET}/{s3_prefix}/")
                upload_model_to_s3(TARGET_MODEL, s3_prefix, dry_run=args.dry_run)

    # Save results
    out = PROJECT_ROOT / "data" / "tokenizer_check_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
