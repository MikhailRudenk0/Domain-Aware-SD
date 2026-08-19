"""
Repair the TurboSparse (Bamboo) target model after loading.

Problem
-------
``modeling_bamboo.py`` was written for transformers 4.41. Its rotary module
registers ``inv_freq`` / ``cos_cached`` / ``sin_cached`` as **non-persistent**
buffers, i.e. they are computed in ``__init__`` and are not part of the
checkpoint. Transformers >= 5 loads remote-code models through a meta-device
path that never re-materializes such buffers, so after
``from_pretrained`` they contain uninitialized memory:

    inv_freq[:5] = [-1.8e+32, 3.3e-41, nan, 4.2e-41, 1.4e-04]
    cos_cached   = all zeros
    sin_cached   = all zeros

RoPE is therefore garbage and the model produces near-uniform logits
(perplexity ~780 on trivial English text, NaN with the sdpa attention path).
This silently invalidated every eval run that used ``target_source=model``.

Fix
---
Recompute ``inv_freq`` from the documented formula and rebuild the cos/sin
caches for every rotary module. Call ``fix_rotary(model)`` right after
``from_pretrained``, or just use ``load_target``.

Verification: ``python -m src.repro.bamboo_fix`` prints perplexity before and
after the repair.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_DIR = PROJECT_ROOT / "TurboSparse-Mistral-Instruct"


def fix_rotary(model: torch.nn.Module, verbose: bool = False) -> int:
    """Recompute inv_freq and the cos/sin caches of every rotary module.

    Returns the number of repaired modules.
    """
    n = 0
    for name, mod in model.named_modules():
        if not hasattr(mod, "inv_freq") or not hasattr(mod, "_set_cos_sin_cache"):
            continue
        dev = mod.inv_freq.device
        dim, base = int(mod.dim), float(mod.base)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=dev).float() / dim))
        mod.register_buffer("inv_freq", inv_freq, persistent=False)
        dtype = getattr(mod, "cos_cached", torch.empty(0)).dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            dtype = torch.get_default_dtype()
        mod._set_cos_sin_cache(seq_len=mod.max_position_embeddings, device=dev, dtype=dtype)
        n += 1
        if verbose and n == 1:
            print(f"  repaired {name}: inv_freq[:3]={inv_freq[:3].tolist()}")
    return n


def load_target(dtype=torch.bfloat16, device: str = "cuda:0", attn: str = "eager"):
    """Load TurboSparse with the rotary buffers repaired."""
    model = AutoModelForCausalLM.from_pretrained(
        str(TARGET_DIR), trust_remote_code=True, dtype=dtype, attn_implementation=attn
    )
    n = fix_rotary(model)
    if n == 0:
        raise RuntimeError("no rotary modules found — did modeling_bamboo.py change?")
    return model.to(device).eval()


def _ppl(model, tok, text: str, device: str) -> tuple[float, float]:
    ids = torch.tensor([tok.encode(text, add_special_tokens=True)]).to(device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0]
    gold = ids[0, 1:]
    acc = float((logits[:-1].argmax(-1) == gold).float().mean())
    nll = torch.nn.functional.cross_entropy(logits[:-1].float(), gold)
    return acc, float(nll.exp())


if __name__ == "__main__":
    from transformers import AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(TARGET_DIR), trust_remote_code=True)
    text = ("The Eiffel Tower is located in Paris, the capital city of France. "
            "It was built in 1889 and remains one of the most visited monuments in the world.")

    model = AutoModelForCausalLM.from_pretrained(
        str(TARGET_DIR), trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device).eval()
    print("before fix: acc=%.3f ppl=%.1f" % _ppl(model, tok, text, device))
    print("repaired modules:", fix_rotary(model, verbose=True))
    print("after  fix: acc=%.3f ppl=%.1f" % _ppl(model, tok, text, device))

    ids = torch.tensor([tok.encode("The capital of France is", add_special_tokens=True)]).to(device)
    with torch.no_grad():
        p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)
    top = torch.topk(p, 5)
    print("'The capital of France is' ->",
          [(tok.decode([int(i)]), round(float(v), 3)) for i, v in zip(top.indices, top.values)])
