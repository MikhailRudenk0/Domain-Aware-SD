# Patches

## `modeling_bamboo_tf5_fix.patch`

Fixes for `TurboSparse-Mistral-Instruct/modeling_bamboo.py` (the model
directory is gitignored, so the fix is kept here as a diff).

Two problems, both caused by the file being written for transformers 4.41
while the environment runs transformers 5.x:

1. **Broken RoPE (the critical one).** transformers >= 5 loads remote-code
   models through a meta-device path that never materializes non-persistent
   buffers, so `inv_freq` / `cos_cached` / `sin_cached` computed in
   `BambooRotaryEmbedding.__init__` contain uninitialized garbage after
   `from_pretrained`. The patch rebuilds them lazily on the first real
   `forward`. This fixes **every** loader that executes the module's forward
   (plain transformers, our scripts, etc.). Note: vLLM still produces garbage
   for this model — it bypasses this code path; do not use vLLM for Bamboo
   without verifying its output by eye.

2. **Cache API drift.** `get_usable_length`, `seen_tokens`, `get_max_length`
   and `from_legacy_cache`/`to_legacy_cache` were removed/renamed in
   transformers 5. The patch adds `hasattr` fallbacks to
   `get_seq_length()` / `get_max_cache_shape()`.

Apply with:

```bash
patch -p0 < patches/modeling_bamboo_tf5_fix.patch
# (run from the project root; target file TurboSparse-Mistral-Instruct/modeling_bamboo.py)
```

The pre-patch original is kept at
`TurboSparse-Mistral-Instruct/modeling_bamboo.py.orig`.

Related: `src/repro/bamboo_fix.py` (`fix_rotary`) repairs an already-loaded
model in-process; both fixes are idempotent and can coexist.
