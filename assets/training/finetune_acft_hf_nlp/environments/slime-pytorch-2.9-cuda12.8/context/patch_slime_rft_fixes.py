# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply Retail RFT compatibility patches to the slime image."""

from __future__ import annotations

import site
import sysconfig
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


def patch_megatron_bridge_optional_import() -> None:
    """Make slime's eager GLM-4.6V bridge import optional.

    ``slime_plugins/megatron_bridge/__init__.py`` unconditionally imports
    ``glm4v_moe``, which imports ``megatron.bridge.models.qwen.qwen_provider`` -
    a module absent from the megatron-bridge build in this image. That raises
    ModuleNotFoundError during MegatronTrainRayActor init. The GLM-4.6V bridge is
    not needed for dense Qwen3-14B, so wrap the import so a missing optional
    dependency is skipped instead of aborting the whole package import.
    """
    path = SLIME_ROOT / "slime_plugins" / "megatron_bridge" / "__init__.py"
    replace_once(
        path,
        "import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge\n",
        "try:\n"
        "    import slime_plugins.megatron_bridge.glm4v_moe  # noqa: F401  # register GLM-4.6V bridge\n"
        "except Exception:  # optional GLM-4.6V bridge; its megatron.bridge deps may be absent\n"
        "    pass\n",
    )


def find_te_common_init() -> Path:
    """Locate transformer_engine/common/__init__.py without importing the package.

    Importing transformer_engine is exactly what triggers the failing assertion,
    so the file is found by scanning the site-packages directories instead.
    """
    candidates: list[Path] = []
    for key in ("purelib", "platlib"):
        try:
            candidates.append(Path(sysconfig.get_paths()[key]))
        except KeyError:
            pass
    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:  # noqa: BLE001 - getsitepackages can be unavailable
        pass

    seen: set[Path] = set()
    for base in candidates:
        if base in seen:
            continue
        seen.add(base)
        cand = base / "transformer_engine" / "common" / "__init__.py"
        if cand.is_file():
            return cand
    raise RuntimeError(
        "Could not locate transformer_engine/common/__init__.py in any "
        f"site-packages directory (searched: {[str(p) for p in seen]})."
    )


def patch_transformer_engine() -> None:
    """Neutralize transformer-engine 2.10's import-time PyPI-metadata checks.

    The ``transformer-engine`` meta package is installed from a platlib wheel
    (``Root-Is-Purelib: false``) while ``transformer-engine-cu12`` is a normal pip
    dist, so ``sanity_checks_for_pypi_installation()`` aborts import with
    ``Could not find `transformer-engine` PyPI package.`` The image also ships
    ``libtransformer_engine.so`` twice (``transformer_engine/`` + ``wheel_lib/``),
    so ``_find_shared_object_in_te_dir`` raises ``Multiple files found``. The
    libraries load fine; only the metadata-only checks are wrong, so neutralize
    them and return the first ``.so`` match instead of raising. The edits cover
    the core and torch/jax extension lookups and the real CUDA/TE library loads
    are preserved.
    """
    path = find_te_common_init()
    # 1. Skip the import-time sanity check (indented inside the runtime-load guard;
    #    the cuDNN/NVRTC/cuRAND/cuBLAS/CUDA-runtime/TE-core loads that follow it in
    #    the same block are preserved).
    replace_once(
        path,
        "\n    sanity_checks_for_pypi_installation()\n",
        "\n    # sanity_checks_for_pypi_installation()  "
        "# AML slime image: bypassed - meta package ships as a platlib wheel\n",
    )
    # 2. Skip the framework-extension version asserts; the torch ext is bundled in
    #    the meta wheel rather than a separate pip dist, and the .so still loads.
    replace_once(
        path,
        "    te_framework_installed = _is_package_installed(module_name)\n",
        "    te_framework_installed = False  "
        "# AML slime image: bypassed - torch ext bundled in meta wheel\n",
    )
    # 3. Tolerate the duplicated core/extension .so (top-level + wheel_lib): return
    #    the first match (search order prefers the source build that matches the
    #    loaded python package) instead of raising.
    replace_once(
        path,
        '    raise RuntimeError(f"Multiple files found: {files}")\n',
        "    return files[0]  "
        "# AML slime image: bypassed - duplicate .so (top-level + wheel_lib)\n",
    )


def main() -> None:
    """Apply all Retail RFT compatibility patches."""
    patch_numpy_guard()
    patch_attention_backend_propagation()
    patch_megatron_bridge_optional_import()
    patch_transformer_engine()


if __name__ == "__main__":
    main()
