"""
Resolve a draft-model directory by name.

Strategy:
    1. If PROJECT_ROOT/<name>/ exists, return it.
    2. Otherwise try to download s3://<bucket>/<prefix>/<name>/ into PROJECT_ROOT/<name>/.
    3. Raise FileNotFoundError if both fail.

Target model loading is the caller's responsibility — by convention the target
is always available locally and never auto-downloaded by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from botocore.exceptions import ClientError

from src.download_from_s3 import download_prefix, list_prefix, make_client

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _local_is_populated(path: Path) -> bool:
    """A model dir counts as 'populated' if it contains any files (not just empty)."""
    if not path.exists() or not path.is_dir():
        return False
    return any(path.rglob("*"))


def resolve_model_dir(
    name: str,
    bucket: str = "domain-aware-sd",
    s3_prefix: str = "models",
    project_root: Optional[Path] = None,
    s3_client=None,
    workers: int = 8,
) -> Path:
    """Return the local directory for the named model, downloading from S3 if needed.

    Parameters
    ----------
    name : Model directory name (e.g. "tiny-mixtral").
    bucket : S3 bucket.
    s3_prefix : S3 key prefix; the model lives at ``<s3_prefix>/<name>/``.
    project_root : Override for testing. Defaults to repo root.
    s3_client : Pre-built boto3 client (injectable for tests). If None, one is
                created lazily — only if the local copy is missing.
    workers : Parallel download threads.

    Returns
    -------
    Local directory path, guaranteed to exist with at least one file inside.

    Raises
    ------
    FileNotFoundError : If the model is not available locally and not on S3.
    """
    root = project_root or PROJECT_ROOT
    local = root / name

    if _local_is_populated(local):
        return local

    client = s3_client or make_client()
    s3_key_prefix = f"{s3_prefix}/{name}"

    try:
        objects = list_prefix(client, s3_key_prefix)
    except ClientError as e:
        raise FileNotFoundError(
            f"Model '{name}' not found locally at {local} and S3 listing failed: {e}"
        ) from e

    if not objects:
        raise FileNotFoundError(
            f"Model '{name}' not found locally at {local} and not present at "
            f"s3://{bucket}/{s3_key_prefix}/"
        )

    print(f"[model_loader] downloading {name} from s3://{bucket}/{s3_key_prefix}/ → {local}")
    # download_prefix reads BUCKET from the module-level env; rely on .env being loaded.
    download_prefix(client, s3_key_prefix, local, force=False, workers=workers, label=name)

    if not _local_is_populated(local):
        raise FileNotFoundError(
            f"Download from s3://{bucket}/{s3_key_prefix}/ produced no files at {local}"
        )

    return local
