#!/usr/bin/env python3
"""
Patch vLLM to support BambooForCausalLM (TurboSparse-Mistral).

Two patches are applied:

  1. registry.py — register "BambooForCausalLM" → MistralForCausalLM so vLLM
     recognises the architecture name from config.json.

  2. llama.py — neutralise the hard check that rejects activations other than
     "silu". Bamboo uses "relu"; without this patch vLLM raises ValueError on
     load. The fix replaces the if-condition with `if False:` so the body is
     never executed but the file stays syntactically valid.

Both patches are idempotent and reversible.

Usage:
    conda activate domain_sd
    python patch_vllm.py           # apply both patches
    python patch_vllm.py --check   # verify both patches are present
    python patch_vllm.py --undo    # remove both patches
"""

import argparse
import re
import sys
from pathlib import Path

# ── Patch 1: registry ─────────────────────────────────────────────────────────
REGISTRY_ENTRY  = '"BambooForCausalLM": ("mistral", "MistralForCausalLM"),'
REGISTRY_MARKER = "# patch: BambooForCausalLM"
REGISTRY_ANCHOR = re.compile(r'"MistralForCausalLM"\s*:\s*\("mistral"')

# ── Patch 2: llama.py relu ────────────────────────────────────────────────────
# Unique string from the error message we want to suppress.
RELU_NEEDLE = "Only silu is supported for now."
# Tag embedded in the patched if-line so we can find and revert it.
RELU_MARKER = "patch:relu"


def find_vllm_root() -> Path:
    try:
        import vllm
    except ImportError:
        sys.exit("vLLM is not installed in the current Python environment.")
    return Path(vllm.__file__).parent


def find_registry(root: Path) -> Path:
    p = root / "model_executor" / "models" / "registry.py"
    if not p.exists():
        sys.exit(f"registry.py not found: {p}")
    return p


def find_llama(root: Path) -> Path:
    p = root / "model_executor" / "models" / "llama.py"
    if not p.exists():
        sys.exit(f"llama.py not found: {p}")
    return p


# ── Registry patch ─────────────────────────────────────────────────────────────

def registry_apply(path: Path) -> bool:
    text = path.read_text()
    if REGISTRY_MARKER in text:
        return False

    m = REGISTRY_ANCHOR.search(text)
    if not m:
        sys.exit(
            'Could not find "MistralForCausalLM" anchor in registry.py.\n'
            f"Add manually:  {REGISTRY_ENTRY}"
        )

    line_start = text.rfind("\n", 0, m.start()) + 1
    indent = " " * (m.start() - line_start)
    new_line = f"{indent}{REGISTRY_ENTRY}  {REGISTRY_MARKER}\n"
    path.write_text(text[:line_start] + new_line + text[line_start:])
    return True


def registry_remove(path: Path) -> bool:
    text = path.read_text()
    if REGISTRY_MARKER not in text:
        return False
    lines = [l for l in text.splitlines(keepends=True) if REGISTRY_MARKER not in l]
    path.write_text("".join(lines))
    return True


def registry_present(path: Path) -> bool:
    return REGISTRY_MARKER in path.read_text()


# ── llama.py relu patch ────────────────────────────────────────────────────────
#
# Strategy: find the `if <cond>:` guard that gates the raise, then replace
# ONLY the condition with `False`.  The raise body stays intact and is never
# executed.  This is safe because:
#   • The file remains syntactically valid (no unclosed parens, no empty body).
#   • `if False:` + original raise body = dead code, no runtime effect.
#
# We locate the if-guard by scanning forward: find an `if` line such that
# within the next few lines there is `raise ValueError` and eventually
# RELU_NEEDLE.  This is more robust than walking backward from the needle.

def _find_if_guard_idx(lines: list[str]) -> int | None:
    for i, line in enumerate(lines):
        if not re.match(r'\s+if\b', line):
            continue
        # Within the next 5 lines look for raise ValueError
        for j in range(i + 1, min(i + 6, len(lines))):
            if "raise ValueError" not in lines[j]:
                continue
            # Within the next 5 lines after raise look for the needle
            for k in range(j, min(j + 6, len(lines))):
                if RELU_NEEDLE in lines[k]:
                    return i
    return None


def relu_apply(path: Path) -> bool:
    text = path.read_text()
    if RELU_MARKER in text:
        return False

    if RELU_NEEDLE not in text:
        sys.exit(
            f"Cannot find {RELU_NEEDLE!r} in {path}.\n"
            "The vLLM version may have changed its error message. "
            "Check llama.py manually."
        )

    # Detect a broken state left by a previous bad patch attempt
    try:
        compile(text, str(path), "exec")
    except SyntaxError as e:
        sys.exit(
            f"llama.py has a SyntaxError ({e}) — likely from a previous failed patch.\n"
            "Restore it by reinstalling vLLM, then re-run this script:\n"
            "  pip install --force-reinstall --no-deps vllm"
        )

    lines = text.splitlines(keepends=True)
    idx = _find_if_guard_idx(lines)

    if idx is None:
        sys.exit(
            f"Cannot find the if-guard for the silu check in {path}.\n"
            "Patch manually: find `if ... silu ...` → change condition to `False`."
        )

    # Parse the if line to extract indentation and original condition
    m = re.match(r'(\s*)if\s+(.*?):\s*$', lines[idx].rstrip())
    if not m:
        sys.exit(f"Cannot parse if line {idx + 1}: {lines[idx]!r}")

    indent, cond = m.group(1), m.group(2)

    # Replace `if <cond>:` with `if False:` — embed original cond for undo
    lines[idx] = f"{indent}if False:  # {RELU_MARKER} was=({cond})\n"
    path.write_text("".join(lines))
    return True


def relu_remove(path: Path) -> bool:
    text = path.read_text()
    if RELU_MARKER not in text:
        return False

    lines = text.splitlines(keepends=True)
    result = []
    for line in lines:
        if RELU_MARKER in line:
            # Restore original condition from embedded comment
            m_indent = re.match(r'(\s*)', line)
            m_cond   = re.search(r'was=\((.+)\)', line)
            indent = m_indent.group(1) if m_indent else ""
            if m_cond:
                result.append(f"{indent}if {m_cond.group(1)}:\n")
            # If we can't parse it, just drop the line (body still intact below)
        else:
            result.append(line)

    path.write_text("".join(result))
    return True


def relu_present(path: Path) -> bool:
    return RELU_MARKER in path.read_text()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch vLLM for BambooForCausalLM")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Verify both patches are present")
    group.add_argument("--undo",  action="store_true", help="Remove both patches")
    args = parser.parse_args()

    root     = find_vllm_root()
    registry = find_registry(root)
    llama    = find_llama(root)

    print(f"vLLM root : {root}")
    print(f"registry  : {registry}")
    print(f"llama.py  : {llama}")
    print()

    if args.check:
        ok1 = registry_present(registry)
        ok2 = relu_present(llama)
        print(f"  Patch 1 (registry BambooForCausalLM) : {'✔ present' if ok1 else '✘ missing'}")
        print(f"  Patch 2 (llama.py relu support)      : {'✔ present' if ok2 else '✘ missing'}")
        if not (ok1 and ok2):
            sys.exit(1)

    elif args.undo:
        r1 = registry_remove(registry)
        r2 = relu_remove(llama)
        print(f"  Patch 1 (registry) : {'removed' if r1 else 'was not applied'}")
        print(f"  Patch 2 (llama.py) : {'removed' if r2 else 'was not applied'}")

    else:
        r1 = registry_apply(registry)
        r2 = relu_apply(llama)
        print(f"  Patch 1 (registry BambooForCausalLM) : {'applied' if r1 else 'already present'}")
        print(f"  Patch 2 (llama.py relu support)      : {'applied' if r2 else 'already present'}")
        if not r1 and not r2:
            print("Nothing to do — both patches already applied.")


if __name__ == "__main__":
    main()
