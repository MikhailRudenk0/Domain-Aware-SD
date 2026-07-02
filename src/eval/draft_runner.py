"""
Draft model runner for teacher-forcing evaluation.

A ``DraftRunner`` wraps a single draft model and exposes a single method,
``forward_batch``, that for each (sample, position) returns:

    argmax_id              int    full-vocab argmax (over draft's softmax)
    own_topk_ids           [Km]   ids of draft's top-Km tokens
    own_topk_probs         [Km]   corresponding probs (renormalized within Km)
    prob_at_target_topk    [Kt]   draft probs at the *target's* top-Kt ids
                                  (Kt may vary per position; values are not
                                  renormalized — caller renormalizes if needed)

Km is fixed (``topk_K`` from config). Kt is per-position (length of the
``target_topk_ids`` provided to this call).

Implementation uses HuggingFace ``transformers`` with a single batched forward
pass — vLLM is intentionally not used here (it is reserved for the data-gen
pipeline only).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from src.eval._torch_utils import resolve_dtype


@dataclass
class DraftPositionInfo:
    argmax_id: int
    own_topk_ids: np.ndarray         # [Km], int64
    own_topk_probs: np.ndarray       # [Km], float32
    prob_at_target_topk: np.ndarray  # [Kt], float32 (Kt may vary)


class DraftRunner:
    def __init__(
        self,
        model_dir: Path,
        device: str,
        dtype: str,
        topk_K: int,
        trust_remote_code: bool = False,
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
        self.vocab_size = self.model.config.vocab_size

    @torch.no_grad()
    def forward_batch(
        self,
        input_ids: torch.Tensor,          # [B, L]
        attention_mask: torch.Tensor,     # [B, L]
        gen_starts: List[int],
        trunk_lens: List[int],
        n_positions: int,
        target_topk_ids_per_sample: List[List[np.ndarray]],
    ) -> List[List[DraftPositionInfo]]:
        """Batched teacher-forcing forward pass.

        Returns a list of length B; element b is a list of length
        ``min(n_positions, trunk_lens[b])`` of per-position info.
        """
        # Target-side trunks may contain ids >= draft vocab (chat special tokens
        # like <|im_end|> emitted by the target). The draft's embedding table
        # cannot look those up, so clamp them to 0 to keep the forward alive.
        # The evaluator is responsible for masking metrics at positions whose
        # context has been clamped (see first_oob_in_trunk in evaluator.py).
        input_ids_safe = input_ids
        if (input_ids >= self.vocab_size).any():
            input_ids_safe = torch.where(
                input_ids < self.vocab_size,
                input_ids,
                torch.zeros_like(input_ids),
            )
        logits = self.model(
            input_ids=input_ids_safe.to(self.device),
            attention_mask=attention_mask.to(self.device),
        ).logits  # [B, L, V_draft]

        B = input_ids.shape[0]
        result: List[List[DraftPositionInfo]] = []
        for b in range(B):
            valid_positions = min(n_positions, trunk_lens[b])
            sample_infos: List[DraftPositionInfo] = []
            for p in range(valid_positions):
                # HF logits[i] predicts position i+1.
                # Trunk position p is at sequence index gen_starts[b] + p,
                # predicted by logits at index gen_starts[b] + p - 1.
                logit_idx = gen_starts[b] + p - 1
                full = torch.softmax(logits[b, logit_idx].float(), dim=-1)  # [V_draft]

                # Draft's own top-Km
                topk = torch.topk(full, k=self.topk_K)
                own_ids = topk.indices.cpu().numpy().astype(np.int64)
                own_probs = topk.values.cpu().numpy().astype(np.float32)

                argmax_id = int(own_ids[0])

                # Look up draft probabilities at target's top-K ids
                target_ids = target_topk_ids_per_sample[b][p]  # [Kt], int64
                if len(target_ids) > 0:
                    # Clamp ids outside draft vocab (handles 32064 vs 32000 case)
                    safe_ids = np.where(
                        (target_ids >= 0) & (target_ids < self.vocab_size),
                        target_ids,
                        0,
                    )
                    probs_at_target = full[torch.as_tensor(safe_ids, device=full.device)]
                    probs_at_target = probs_at_target.cpu().numpy().astype(np.float32)
                    # Zero out positions where the target id was outside vocab
                    out_of_vocab = (target_ids < 0) | (target_ids >= self.vocab_size)
                    if out_of_vocab.any():
                        probs_at_target = probs_at_target.copy()
                        probs_at_target[out_of_vocab] = 0.0
                else:
                    probs_at_target = np.zeros(0, dtype=np.float32)

                sample_infos.append(
                    DraftPositionInfo(
                        argmax_id=argmax_id,
                        own_topk_ids=own_ids,
                        own_topk_probs=own_probs,
                        prob_at_target_topk=probs_at_target,
                    )
                )
            result.append(sample_infos)
        return result

    def free(self) -> None:
        """Best-effort release of GPU memory."""
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
