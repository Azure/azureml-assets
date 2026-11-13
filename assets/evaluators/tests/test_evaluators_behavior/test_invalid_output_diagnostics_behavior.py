# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Guard that builtin evaluators report invalid judge output through the diagnostic helper."""

from pathlib import Path

import pytest

_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "builtin"


@pytest.mark.unittest
def test_all_invalid_output_raise_sites_use_the_diagnostic_helper():
    """No builtin evaluator raises the generic, cause-less invalid output message."""
    generic = 'message="Evaluator returned invalid output."'
    offenders = [
        str(path.relative_to(_BUILTIN_DIR))
        for path in _BUILTIN_DIR.glob("*/evaluator/_*.py")
        if generic in path.read_text(encoding="utf-8")
    ]
    assert not offenders
