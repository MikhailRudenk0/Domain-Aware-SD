#!/usr/bin/env python3
"""
Patch vLLM to support BambooForCausalLM (TurboSparse-Mistral).

Two patches are applied:

  1. registry.py — register "BambooForCausalLM" → MistralForCausalLM so vLLM
     recognises the architecture name from config.json.

  2. llama.py — remove the hard ValueError that rejects any activation other
     than "silu". Bamboo uses "relu" in its config; vLLM's LlamaMLP raises on
     load without this patch. After patching, relu (and any other activation
     present in PyTorch's ACT2FN table) is accepted silently.

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

# ── Patch 1: registry ────────────────────────────────────────────────────────
REGISTRY_ENTRY  = '"BambooForCausalLM": ("mistral", "MistralForCausalLM"),'
REGISTRY_MARKER = "# patch: BambooForCausalLM"
REGISTRY_ANCHOR = re.compile(r'"MistralForCausalLM"\s*:\s*\("mistral"')

# ── Patch 2: llama.py relu ────────────────────────────────────────────────────
# vLLM's LlamaMLP raises ValueError when hidden_act != "silu".
# We locate the raise by the unique substring in its message and comment it out.
RELU_MARKER   = "# patch: allow non-silu activations (relu for Bamboo)"
RELU_NEEDLE   = "Only silu is supported for now."   # unique string in the raise message


def find_vllm_root() -> Path:
    try:
        import vllm
    except ImportError:
        sys.exit("vLLM is not installed in the current Python environment.")
    return Path(vllm.__file__).parent


def find_registry(root: Path) -> Path:
    p = root / "model_executor" / "models" / "registry.py"
    if not p.exists():
        sys.exit(f"Registry not found: {p}")
    return p


def find_llama(root: Path) -> Path:
    p = root / "model_executor" / "models" / "llama.py"
    if not p.exists():
        sys.exit(f"llama.py not found: {p}")
    return p


# ── Registry patch ────────────────────────────────────────────────────────────

def registry_apply(path: Path) -> bool:
    text = path.read_text()
    if REGISTRY_MARKER in text:
        return False  # already applied

    m = REGISTRY_ANCHOR.search(text)
    if not m:
        sys.exit(
            'Could not find "MistralForCausalLM" anchor in registry.py.\n'
            "The vLLM version may use a different registry format.\n"
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


# ── llama.py relu patch ───────────────────────────────────────────────────────

def relu_apply(path: Path) -> bool:
    """
    Find the raise ValueError block that contains RELU_NEEDLE and comment it out.
    Handles both single-line and multi-line raise statements.
    """
    text = path.read_text()
    if RELU_MARKER in text:
        return False  # already applied
    if RELU_NEEDLE not in text:
        sys.exit(
            f"Could not find the silu-only check in llama.py (needle: {RELU_NEEDLE!r}).\n"
            "The vLLM version may have changed. Check the file manually:\n"
            f"  {path}"
        )

    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    patched = False
    while i < len(lines):
        line = lines[i]
        if RELU_NEEDLE in line:
            # Walk backward to find the start of the raise / if block
            # Find the raise statement that contains or precedes this line
            raise_start = i
            while raise_start > 0 and "raise" not in lines[raise_start]:
                raise_start -= 1

            # Also comment out any single-line `if` that guards this raise
            if_start = raise_start
            prev = raise_start - 1
            if prev >= 0 and re.match(r'\s*if\b', lines[prev]):
                if_start = prev

            # Find the end of the raise statement (closing paren)
            raise_end = raise_start
            open_parens = 0
            for j in range(raise_start, len(lines)):
                open_parens += lines[j].count("(") - lines[j].count(")")
                raise_end = j
                if open_parens <= 0:
                    break

            # Comment out lines from if_start to raise_end (inclusive)
            indent = re.match(r'(\s*)', lines[if_start]).group(1)
            marker_line = f"{indent}{RELU_MARKER}\n"
            out.append(marker_line)
            for j in range(if_start, raise_end + 1):
                out.append(indent + "# " + lines[j].lstrip())
            i = raise_end + 1
            patched = True
        else:
            out.append(line)
            i += 1

    if not patched:
        sys.exit("Failed to locate and patch the raise block in llama.py.")

    path.write_text("".join(out))
    return True


def relu_remove(path: Path) -> bool:
    text = path.read_text()
    if RELU_MARKER not in text:
        return False

    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        if RELU_MARKER in lines[i]:
            # Skip the marker line and all following comment lines that belong to the block
            i += 1
            while i < len(lines) and re.match(r'\s*#', lines[i]):
                # Stop if this comment is something unrelated (doesn't look like commented code)
                if RELU_NEEDLE in lines[i] or "raise" in lines[i] or "if " in lines[i]:
                    i += 1
                else:
                    break
        else:
            out.append(lines[i])
            i += 1

    path.write_text("".join(out))
    return True


def relu_present(path: Path) -> bool:
    return RELU_MARKER in path.read_text()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch vLLM for BambooForCausalLM")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check if patches are present")
    group.add_argument("--undo",  action="store_true", help="Remove patches")
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
