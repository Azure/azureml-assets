# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply SGLang CVE-2026-5760 template sandboxing fixes."""

from __future__ import annotations

import site
import sysconfig
from pathlib import Path


def site_package_roots() -> list[Path]:
    """Return Python package roots without importing sglang."""
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.append(Path(path))
    roots.extend(Path(path) for path in site.getsitepackages())
    seen: set[Path] = set()
    return [path for path in roots if not (path in seen or seen.add(path))]


def find_sglang_root() -> Path:
    """Locate the installed sglang package."""
    for root in site_package_roots():
        candidate = root / "sglang"
        if candidate.is_dir():
            return candidate
    raise RuntimeError("Could not locate installed sglang package")


def replace_once(path: Path, old: str, new: str) -> None:
    """Replace one expected text block in a file."""
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_tiktoken_tokenizer(sglang_root: Path) -> None:
    """Use a Jinja sandbox for tiktoken chat templates."""
    path = sglang_root / "srt" / "tokenizer" / "tiktoken_tokenizer.py"
    replace_once(
        path,
        "        from jinja2 import Template\n",
        "        from jinja2.sandbox import ImmutableSandboxedEnvironment\n",
    )
    replace_once(
        path,
        "        self.chat_template_jinja = Template(self.chat_template)\n",
        "        self.chat_template_jinja = ImmutableSandboxedEnvironment(loader=None).from_string(\n"
        "            self.chat_template\n"
        "        )\n",
    )


def main() -> None:
    """Apply all SGLang security patches."""
    patch_tiktoken_tokenizer(find_sglang_root())


if __name__ == "__main__":
    main()
