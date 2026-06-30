"""
Target-distribution providers.

Two implementations:

* ``DatasetTargetProvider`` — reads target's top-K (ids + probs) from
  ``SpecDecDataset`` records produced by ``generate_synthetic_data.py``.
  Padding entries (prob == 0) are stripped; positions with no real entries
  are reported as skipped.

* ``ModelTargetProvider`` — runs the target model in teacher-forcing mode and
  extracts a top-K view of the full output distribution.

Both return ``TargetInfo`` per (sample, position):
    topk_ids       [K]   token ids of target's top-K
    topk_probs     [K]   probabilities, renormalized to sum to 1 within K
    is_skipped     bool  True if the position has no useful target signal
                         (only possible for ``DatasetTargetProvider``)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.eval._torch_utils import resolve_dtype


@dataclass
class TargetInfo:
    topk_ids: np.ndarray          # [K], int64
    topk_probs: np.ndarray        # [K], float32, renormalized
    is_skipped: bool

    @classmethod
    def skipped(cls) -> "TargetInfo":
        return cls(
            topk_ids=np.zeros(0, dtype=np.int64),
            topk_probs=np.zeros(0, dtype=np.float32),
            is_skipped=True,
        )


class DatasetTargetProvider:
    """Target-info source backed by the synthetic dataset's top-K entries.

    Strips padding rows (id == 0, prob == 0). Renormalizes the kept probs
    so they sum to 1 within the K support.
    """

    def get_for_sample(self, sample: dict) -> List[TargetInfo]:
        if "top10_ids" not in sample or "top10_probs" not in sample:
            raise RuntimeError(
                "DatasetTargetProvider requires SpecDecDataset records in "
                "distillation mode (top10_ids / top10_probs present)."
            )
        ids_t: torch.Tensor = sample["top10_ids"]      # [gen_len, 10]
        probs_t: torch.Tensor = sample["top10_probs"]  # [gen_len, 10]
        ids = ids_t.numpy()
        probs = probs_t.numpy()

        infos: List[TargetInfo] = []
        for p in range(ids.shape[0]):
            pos_probs = probs[p]
            # Skipped iff all probs are zero (matches the empty-row encoding
            # used by generate_synthetic_data.py + dataset padding).
            if float(pos_probs.sum()) <= 1e-9:
                infos.append(TargetInfo.skipped())
                continue
            mask = pos_probs > 0
            pos_ids = ids[p][mask].astype(np.int64)
            pos_probs = pos_probs[mask].astype(np.float32)
            pos_probs = pos_probs / pos_probs.sum()
            infos.append(TargetInfo(pos_ids, pos_probs, False))
        return infos


class ModelTargetProvider:
    """Target-info source backed by a transformers causal LM forward pass.

    Builds a top-K view of the full output distribution at each evaluated
    trunk position. K is configured via ``topk_K``. Renormalizes within K
    so the K support sums to 1 (which is what the metrics expect).
    """

    def __init__(
        self,
        model_dir: Path,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        topk_K: int,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = resolve_dtype(dtype, device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        ).to(device).eval()
        self.device = device
        self.topk_K = topk_K

    @torch.no_grad()
    def get_for_batch(
        self,
        input_ids: torch.Tensor,           # [B, L]
        attention_mask: torch.Tensor,      # [B, L]
        gen_starts: List[int],
        trunk_lens: List[int],
        n_positions: int,
    ) -> List[List[TargetInfo]]:
        logits = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
        ).logits  # [B, L, V]

        B = input_ids.shape[0]
        result: List[List[TargetInfo]] = []
        for b in range(B):
            valid_positions = min(n_positions, trunk_lens[b])
            sample_infos: List[TargetInfo] = []
            for p in range(valid_positions):
                # HF logits[i] predicts token at position i+1.
                # Trunk position p is at sequence index gen_start[b] + p; it is
                # predicted by logits at index gen_start[b] + p - 1.
                logit_idx = gen_starts[b] + p - 1
                full = torch.softmax(logits[b, logit_idx].float(), dim=-1)  # [V]
                topk = torch.topk(full, k=self.topk_K)
                ids = topk.indices.cpu().numpy().astype(np.int64)
                probs = topk.values.cpu().numpy().astype(np.float32)
                s = float(probs.sum())
                if s > 0:
                    probs = probs / s
                sample_infos.append(TargetInfo(ids, probs, False))
            result.append(sample_infos)
        return result

    def free(self) -> None:
        """Best-effort release of GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
