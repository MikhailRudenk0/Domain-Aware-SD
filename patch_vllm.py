#!/usr/bin/env python3
"""
Patch vLLM's model registry to recognise BambooForCausalLM (TurboSparse-Mistral).

vLLM only loads architectures in its own registry. BambooForCausalLM is Mistral-7B
with a sparse MLP predictor; all standard weight names/shapes are identical to Mistral.
Mapping it to MistralForCausalLM lets vLLM load the matching weights (predictor weights
are silently skipped) and run the model correctly for inference / logprob capture.

Usage (run once after installing / upgrading vLLM):
    conda activate domain_sd
    python patch_vllm.py          # apply patch
    python patch_vllm.py --check  # verify patch is present
    python patch_vllm.py --undo   # remove patch
"""

import argparse
import re
import sys
from pathlib import Path

ENTRY      = '"BambooForCausalLM": ("mistral", "MistralForCausalLM"),'
MARKER     = "# patch: BambooForCausalLM"
ANCHOR_PAT = re.compile(r'"MistralForCausalLM"\s*:\s*\("mistral"')


def find_registry() -> Path:
    try:
        import vllm
    except ImportError:
        sys.exit("vLLM is not installed in the current Python environment.")

    candidate = Path(vllm.__file__).parent / "model_executor" / "models" / "registry.py"
    if not candidate.exists():
        sys.exit(f"Registry file not found at expected path:\n  {candidate}")
    return candidate


def apply(registry: Path) -> None:
    text = registry.read_text()

    if MARKER in text:
        print(f"Patch already applied — nothing to do.\n  {registry}")
        return

    match = ANCHOR_PAT.search(text)
    if not match:
        sys.exit(
            'Could not find "MistralForCausalLM" anchor in registry. '
            "The vLLM version may have changed its registry format. "
            "Add the entry manually:\n"
            f"  {ENTRY}"
        )

    # Insert our entry on the line immediately before the Mistral entry
    insert_at = match.start()
    line_start = text.rfind("\n", 0, insert_at) + 1
    indent = " " * (insert_at - line_start)
    new_line = f"{indent}{ENTRY}  {MARKER}\n"
    patched = text[:line_start] + new_line + text[line_start:]

    registry.write_text(patched)
    print(f"Patch applied.\n  {registry}")
    print(f"  Added: {ENTRY}")


def check(registry: Path) -> None:
    text = registry.read_text()
    if MARKER in text:
        print(f"Patch is present.\n  {registry}")
    else:
        print(f"Patch is NOT present.\n  {registry}")
        sys.exit(1)


def undo(registry: Path) -> None:
    text = registry.read_text()
    if MARKER not in text:
        print("Patch not found — nothing to remove.")
        return

    lines = text.splitlines(keepends=True)
    patched = "".join(l for l in lines if MARKER not in l)
    registry.write_text(patched)
    print(f"Patch removed.\n  {registry}")


def main():
    parser = argparse.ArgumentParser(description="Patch vLLM registry for BambooForCausalLM")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="Check if patch is present")
    group.add_argument("--undo",  action="store_true", help="Remove the patch")
    args = parser.parse_args()

    registry = find_registry()
    print(f"Registry: {registry}")

    if args.check:
        check(registry)
    elif args.undo:
        undo(registry)
    else:
        apply(registry)


if __name__ == "__main__":
    main()
