# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Defer unrelated AutoML entrypoints when importing NLP logging utilities."""

import ast
from pathlib import Path
import sysconfig


def patch_entrypoints(path):
    """Preserve entrypoint module exports without eagerly importing their dependencies."""
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    imports = {}
    removed_lines = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            for alias in node.names:
                if alias.asname is None or alias.name == "*":
                    raise RuntimeError("Unexpected AutoML entrypoint import.")
                imports[alias.asname] = "." + node.module + "." + alias.name
            removed_lines.update(range(node.lineno - 1, node.end_lineno))
    if not {"entrypoint_util", "training_entrypoint_util"} <= imports.keys():
        raise RuntimeError("AutoML entrypoint imports changed; review the compatibility backport.")
    loader = (
        "from importlib import import_module as _import_module\n\n"
        f"_ENTRYPOINT_IMPORTS = {imports!r}\n"
        "__all__ = list(_ENTRYPOINT_IMPORTS)\n\n\n"
        "def __getattr__(name):\n"
        "    if name not in _ENTRYPOINT_IMPORTS:\n"
        "        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')\n"
        "    module = _import_module(_ENTRYPOINT_IMPORTS[name], __name__)\n"
        "    globals()[name] = module\n"
        "    return module\n\n\n"
    )
    patched = loader + "".join(line for i, line in enumerate(lines) if i not in removed_lines)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    patch_entrypoints(
        Path(sysconfig.get_path("purelib"))
        / "azureml" / "train" / "automl" / "runtime" / "_entrypoints" / "__init__.py"
    )
