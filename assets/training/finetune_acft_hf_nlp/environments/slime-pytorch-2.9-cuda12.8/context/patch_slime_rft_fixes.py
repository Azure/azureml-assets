# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply Retail RFT compatibility patches to the slime image."""

from __future__ import annotations

from pathlib import Path


SLIME_ROOT = Path("/opt/slime")


def replace_once(path: Path, old: str, new: str) -> None:
    """Replace one expected text block in a file."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_numpy_guard() -> None:
    """Remove slime's conservative numpy 1.x startup assertion."""
    path = SLIME_ROOT / "slime" / "backends" / "megatron_utils" / "initialize.py"
    replace_once(
        path,
        '    assert np.__version__.startswith("1."), "Megatron does not support numpy 2.x"\n',
        "    # This image builds Megatron extensions against numpy 2.x, so the\n"
        "    # conservative slime numpy 1.x guard is not needed here.\n",
    )


def patch_attention_backend_propagation() -> None:
    """Propagate CLI attention backend selection into Megatron Bridge config."""
    path = SLIME_ROOT / "slime" / "backends" / "megatron_utils" / "model_provider.py"
    text = path.read_text(encoding="utf-8")
    marker = "        provider.context_parallel_size = args.context_parallel_size\n"
    patch = (
        marker
        +
        "        if hasattr(args, \"attention_backend\") and args.attention_backend is not None:\n"
        "            provider.attention_backend = args.attention_backend\n"
    )
    if patch in text:
        return
    if marker not in text:
        raise RuntimeError(f"Expected provider marker not found in {path}")
    path.write_text(text.replace(marker, patch, 1), encoding="utf-8")


def main() -> None:
    """Apply all Retail RFT compatibility patches."""
    patch_numpy_guard()
    patch_attention_backend_propagation()


if __name__ == "__main__":
    main()
