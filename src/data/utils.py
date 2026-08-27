from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np


def walk_data_files(root: Union[str, Path]) -> List[Path]:
    """Recursively collect all *.jsonl and *.npz files under root, sorted."""
    root = Path(root)
    return sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.npz")))


# Back-compat alias — older callers imported walk_jsonl directly.
def walk_jsonl(root: Union[str, Path]) -> List[Path]:
    return walk_data_files(root)


def detect_format(record: dict) -> str:
    """
    Detect record format from field names.
    Returns 'synthetic', 'flan', or 'plain'.
    For NpzRecord objects, always returns 'synthetic'.
    """
    if isinstance(record, NpzRecord):
        return "synthetic"
    if "trunk" in record and ("top10_ids" in record or "top10" in record):
        return "synthetic"
    if "inputs" in record and "targets" in record:
        return "flan"
    if "text" in record:
        return "plain"
    raise ValueError(f"Unrecognized record format. Keys: {list(record.keys())}")


def _normalize_top10(record: dict) -> None:
    """
    In-place: convert legacy top10 (list of list of dicts) to the new schema
    (parallel top10_ids / top10_probs arrays). No-op if already normalized.
    Drops the "token" field — it's unused downstream.
    """
    if "top10_ids" in record:
        return
    legacy = record.pop("top10", None)
    if legacy is None:
        return
    ids: list[list[int]] = []
    probs: list[list[float]] = []
    for pos in legacy:
        if not pos:
            ids.append([])
            probs.append([])
            continue
        ids.append([int(e["token_id"]) for e in pos])
        probs.append([float(e["prob"]) for e in pos])
    record["top10_ids"] = ids
    record["top10_probs"] = probs


class NpzRecord:
    """Memory-efficient record that references numpy arrays directly instead
    of converting to Python lists. Acts like a dict for backward compatibility."""

    __slots__ = ("_store", "_trunk_offset", "_trunk_len",
                 "_all_trunk_ids", "_all_top10_ids", "_all_top10_probs_q",
                 "_all_top10_mask", "_k")

    def __init__(
        self,
        cluster: str,
        prompt: str,
        reference: str,
        trunk_offset: int,
        trunk_len: int,
        all_trunk_ids: np.ndarray,
        all_top10_ids: np.ndarray,
        all_top10_probs_q: np.ndarray,
        all_top10_mask: np.ndarray,
        k: int,
    ):
        self._store = {"cluster": cluster, "prompt": prompt, "reference": reference}
        self._trunk_offset = trunk_offset
        self._trunk_len = trunk_len
        self._all_trunk_ids = all_trunk_ids
        self._all_top10_ids = all_top10_ids
        self._all_top10_probs_q = all_top10_probs_q
        self._all_top10_mask = all_top10_mask
        self._k = k

    def __contains__(self, key):
        return key in ("cluster", "prompt", "reference", "trunk", "top10_ids", "top10_probs")

    def __getitem__(self, key):
        return self.get(key)

    def get(self, key, default=None):
        if key in self._store:
            return self._store[key]

        o = self._trunk_offset
        n = self._trunk_len

        if key == "trunk":
            return self._all_trunk_ids[o:o + n].astype(int).tolist()

        if key == "top10_ids":
            if self._k <= 0 or n <= 0:
                return [[] for _ in range(n)]
            ids_slice = self._all_top10_ids[o:o + n]
            mask_slice = self._all_top10_mask[o:o + n]
            result = []
            for j in range(n):
                if mask_slice[j]:
                    result.append(ids_slice[j].astype(int).tolist())
                else:
                    result.append([])
            return result

        if key == "top10_probs":
            if self._k <= 0 or n <= 0:
                return [[] for _ in range(n)]
            probs_slice = self._all_top10_probs_q[o:o + n]
            mask_slice = self._all_top10_mask[o:o + n]
            result = []
            for j in range(n):
                if mask_slice[j]:
                    result.append((probs_slice[j].astype(float) / 1000.0).tolist())
                else:
                    result.append([])
            return result

        return default

    def keys(self):
        return ["cluster", "prompt", "reference", "trunk", "top10_ids", "top10_probs"]

    @property
    def trunk_len(self) -> int:
        return self._trunk_len

    def mean_top1_prob(self) -> float | None:
        """Compute mean top-1 probability without materializing Python lists."""
        n = self._trunk_len
        if self._k <= 0 or n <= 0:
            return None
        o = self._trunk_offset
        mask_slice = self._all_top10_mask[o:o + n]
        probs_slice = self._all_top10_probs_q[o:o + n]
        valid_mask = mask_slice.astype(bool)
        if not valid_mask.any():
            return None
        top1_vals = probs_slice[valid_mask, 0].astype(float) / 1000.0
        return float(top1_vals.mean())


def _load_jsonl(path: Path) -> List[dict]:
    records: List[dict] = []
    default_cluster = path.stem
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "cluster" not in record:
                record["cluster"] = default_cluster
            _normalize_top10(record)
            records.append(record)
    return records


def _load_npz(path: Path) -> List:
    """
    Load npz-packed cluster into a list of NpzRecord objects that lazily
    reference the underlying numpy arrays. This avoids the ~20x memory blowup
    of converting every row to Python lists upfront.
    """
    archive = np.load(path, allow_pickle=True)
    default_cluster = str(archive["cluster"]) if "cluster" in archive.files else path.stem
    prompts = archive["prompts"]
    references = archive["references"] if "references" in archive.files else None
    trunk_lens = archive["trunk_lens"].astype(int)
    trunk_ids = archive["trunk_ids"]
    top10_ids_all = archive["top10_ids"]
    top10_probs_q = archive["top10_probs_q"]
    top10_mask = archive["top10_mask"]
    k = int(archive["top_k"]) if "top_k" in archive.files else top10_ids_all.shape[1]

    records = []
    offset = 0
    for i, n in enumerate(trunk_lens):
        n = int(n)
        records.append(NpzRecord(
            cluster=default_cluster,
            prompt=str(prompts[i]),
            reference=str(references[i]) if references is not None else "",
            trunk_offset=offset,
            trunk_len=n,
            all_trunk_ids=trunk_ids,
            all_top10_ids=top10_ids_all,
            all_top10_probs_q=top10_probs_q,
            all_top10_mask=top10_mask,
            k=k,
        ))
        offset += n

    return records


def build_index(paths: List[Path]) -> Tuple[List, List[str]]:
    """
    Load all records from a list of JSONL or NPZ paths into memory.
    Returns (records, cluster_labels).
    Cluster label comes from record['cluster'] if present, else from the file stem.
    """
    records = []
    cluster_labels: List[str] = []

    for path in paths:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            loaded = _load_npz(path)
        else:
            loaded = _load_jsonl(path)
        for record in loaded:
            records.append(record)
            cluster_labels.append(record.get("cluster", path.stem))

    return records, cluster_labels
