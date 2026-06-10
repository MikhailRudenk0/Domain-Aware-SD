from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset, Subset

from .utils import build_index, detect_format, walk_jsonl

_MODE = Literal["distillation", "standard"]


class SpecDecDataset(Dataset):
    """
    Dataset for speculative decoding draft-model training and evaluation.

    Supports three source formats (auto-detected per record):
      synthetic — has 'trunk' + 'top10' fields (output of generate_synthetic_data.py)
      flan      — has 'inputs' + 'targets' fields (raw Flan JSONL)
      plain     — has 'text' field

    In 'distillation' mode, __getitem__ also returns top10_ids / top10_probs —
    a sparse [gen_len, 10] representation of the target distribution.
    Probabilities are rounded to 3 decimal places.

    In 'standard' mode, only input_ids / attention_mask / labels are returned.
    """

    def __init__(
        self,
        paths: Union[str, Path, Sequence[Union[str, Path]]],
        tokenizer,
        mode: _MODE = "distillation",
        max_length: int = 512,
        max_gen_length: int = 256,
        clusters_filter: Optional[List[str]] = None,
        min_top1_prob: Optional[float] = None,
    ) -> None:
        super().__init__()

        if isinstance(paths, (str, Path)):
            paths = [Path(paths)]
        else:
            paths = [Path(p) for p in paths]

        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.max_gen_length = max_gen_length

        records, cluster_labels = build_index(paths)

        if clusters_filter is not None:
            cluster_set = set(clusters_filter)
            pairs = [(r, c) for r, c in zip(records, cluster_labels) if c in cluster_set]
            records = [p[0] for p in pairs]
            cluster_labels = [p[1] for p in pairs]

        if min_top1_prob is not None:
            kept_r, kept_c = [], []
            for r, c in zip(records, cluster_labels):
                fmt = detect_format(r)
                if fmt == "synthetic" and r.get("top10"):
                    top1_probs = [pos[0]["prob"] for pos in r["top10"] if pos]
                    if top1_probs and (sum(top1_probs) / len(top1_probs)) < min_top1_prob:
                        continue
                kept_r.append(r)
                kept_c.append(c)
            records, cluster_labels = kept_r, kept_c

        self._records = records
        self._clusters = cluster_labels

        self._cluster_to_indices: Dict[str, List[int]] = defaultdict(list)
        for i, c in enumerate(cluster_labels):
            self._cluster_to_indices[c].append(i)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(cls, root: Union[str, Path], **kwargs) -> "SpecDecDataset":
        """Walk root recursively, collect all *.jsonl files, return a dataset."""
        paths = walk_jsonl(root)
        if not paths:
            raise FileNotFoundError(f"No *.jsonl files found under {root}")
        return cls(paths, **kwargs)

    # ------------------------------------------------------------------
    # Core Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict:
        record = self._records[idx]
        cluster = self._clusters[idx]
        fmt = detect_format(record)

        # Build prompt_ids, trunk_ids, and top10 per format
        if fmt == "synthetic":
            prompt_ids = self.tokenizer.encode(
                record["prompt"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 1,
            )
            trunk_ids: List[int] = record["trunk"]
            top10 = record.get("top10", [])

        elif fmt == "flan":
            prompt_ids = self.tokenizer.encode(
                record["inputs"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length - 1,
            )
            trunk_ids = self.tokenizer.encode(
                record["targets"],
                add_special_tokens=False,
            )
            top10 = []

        else:  # plain — whole text is one causal sequence, no separate prompt
            trunk_ids = self.tokenizer.encode(
                record["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )
            prompt_ids = []
            top10 = []

        # Truncate trunk to fit within max_length and max_gen_length
        available = self.max_length - len(prompt_ids)
        trunk_ids = trunk_ids[: min(available, self.max_gen_length)]
        top10 = top10[: len(trunk_ids)]

        gen_start = len(prompt_ids)
        full_ids = prompt_ids + trunk_ids
        seq_len = len(full_ids)

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(seq_len, dtype=torch.long)

        # -100 masks prompt positions from the loss; trunk positions carry real labels.
        # HuggingFace causal LM models apply an internal shift (logits[i] → labels[i+1]),
        # so labels[gen_start] = trunk_ids[0] means the last prompt token predicts the
        # first generated token — exactly what we want.
        labels = torch.full((seq_len,), -100, dtype=torch.long)
        if trunk_ids:
            labels[gen_start:] = torch.tensor(trunk_ids, dtype=torch.long)

        item: dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "gen_start": gen_start,
            "cluster": cluster,
        }

        if self.mode == "distillation" and top10:
            n = 10
            ids_list, probs_list = [], []
            for pos in top10:
                entries = pos[:n]
                pos_ids = [e["token_id"] for e in entries]
                pos_probs = [round(e["prob"], 3) for e in entries]
                # Pad to exactly n if the position has fewer than 10 candidates
                pad = n - len(pos_ids)
                pos_ids += [0] * pad
                pos_probs += [0.0] * pad
                ids_list.append(pos_ids)
                probs_list.append(pos_probs)

            item["top10_ids"] = torch.tensor(ids_list, dtype=torch.long)       # [gen_len, 10]
            item["top10_probs"] = torch.tensor(probs_list, dtype=torch.float32) # [gen_len, 10]

        return item

    # ------------------------------------------------------------------
    # Cluster utilities
    # ------------------------------------------------------------------

    def cluster_names(self) -> List[str]:
        """Return sorted list of cluster names present in the dataset."""
        return sorted(self._cluster_to_indices.keys())

    def get_cluster_subset(self, name: str) -> Subset:
        """Return a Subset containing only samples from the named cluster."""
        indices = self._cluster_to_indices.get(name, [])
        if not indices:
            raise KeyError(f"Cluster '{name}' not found. Available: {self.cluster_names()}")
        return Subset(self, indices)

    # ------------------------------------------------------------------
    # Train / val / test split
    # ------------------------------------------------------------------

    def split(
        self,
        ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Stratified split by cluster so each split has proportional representation.
        Returns (train_subset, val_subset, test_subset).
        """
        if abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios):.4f}")

        train_r, val_r, _ = ratios
        rng = random.Random(seed)

        train_idx, val_idx, test_idx = [], [], []
        for indices in self._cluster_to_indices.values():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_r)
            n_val = int(n * val_r)
            train_idx.extend(shuffled[:n_train])
            val_idx.extend(shuffled[n_train: n_train + n_val])
            test_idx.extend(shuffled[n_train + n_val:])

        return Subset(self, train_idx), Subset(self, val_idx), Subset(self, test_idx)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """
        Compute summary statistics over the dataset.

        Returns:
          total_samples, num_clusters, cluster_counts,
          gen_len (mean/min/max) — synthetic records only,
          mean_top1_prob         — synthetic records only.
        """
        cluster_counts = {c: len(idx) for c, idx in self._cluster_to_indices.items()}

        gen_lens: List[int] = []
        top1_probs: List[float] = []

        for rec in self._records:
            fmt = detect_format(rec)
            if fmt == "synthetic":
                gen_lens.append(len(rec.get("trunk", [])))
                positions = rec.get("top10", [])
                if positions:
                    mean_p = sum(pos[0]["prob"] for pos in positions if pos) / len(positions)
                    top1_probs.append(mean_p)

        result: dict = {
            "total_samples": len(self._records),
            "num_clusters": len(cluster_counts),
            "cluster_counts": cluster_counts,
        }

        if gen_lens:
            result["gen_len"] = {
                "mean": round(sum(gen_lens) / len(gen_lens), 1),
                "min": min(gen_lens),
                "max": max(gen_lens),
            }

        if top1_probs:
            result["mean_top1_prob"] = round(sum(top1_probs) / len(top1_probs), 3)

        return result
