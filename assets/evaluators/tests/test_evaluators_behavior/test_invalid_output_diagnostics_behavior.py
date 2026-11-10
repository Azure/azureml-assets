# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the privacy-safe invalid-output diagnostics shared by builtin evaluators."""

import importlib
from pathlib import Path

import pytest

_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "builtin"
_EVALUATOR_MODULES = sorted(
    f"{path.parents[1].name}.evaluator.{path.stem}"
    for path in _BUILTIN_DIR.glob("*/evaluator/_*.py")
    if "def _invalid_output_message(" in path.read_text(encoding="utf-8")
)

_SECRET = "CUSTOMER-SECRET-TEXT"

_CASES = [
    (None, "returned an empty response", "finish_reason=unknown, output_tokens=unknown"),
    ({"llm_output": "", "finish_reason": "stop", "output_token_count": 0}, "returned an empty response",
     "finish_reason=stop, output_tokens=0"),
    ({"llm_output": f'{{"score": 4, "reason": "{_SECRET}', "finish_reason": "length", "output_token_count": 800},
     "output was truncated", "finish_reason=length, output_tokens=800"),
    ({"llm_output": _SECRET, "finish_reason": "content_filter", "output_token_count": 3},
     "blocked by the content filter", "finish_reason=content_filter"),
    ({"llm_output": _SECRET, "finish_reason": "stop", "output_token_count": 5}, "was not a valid JSON object",
     "finish_reason=stop, output_tokens=5"),
    ({"llm_output": [_SECRET], "finish_reason": "stop"}, "had unexpected type list", "finish_reason=stop"),
    # Raw non-dict flow output is diagnosed by its own type, not reported as empty.
    (_SECRET, "was not a valid JSON object", "finish_reason=unknown, output_tokens=unknown"),
    ([_SECRET], "had unexpected type list", "finish_reason=unknown"),
    # Unexpected metadata values are never echoed.
    ({"llm_output": "", "finish_reason": _SECRET, "output_token_count": {"content": _SECRET}},
     "returned an empty response", "finish_reason=unknown, output_tokens=unknown"),
    # The _do_eval NaN check passes the result properties, which carry the token metadata.
    ({"reason": _SECRET, "finish_reason": "stop", "completion_tokens": 12}, "no valid score could be parsed",
     "finish_reason=stop, output_tokens=12"),
]


@pytest.mark.unittest
def test_all_invalid_output_raise_sites_use_the_diagnostic_helper():
    """Every builtin evaluator reports invalid output through the diagnostic helper."""
    generic = 'message="Evaluator returned invalid output."'
    offenders = [
        str(path.relative_to(_BUILTIN_DIR))
        for path in _BUILTIN_DIR.glob("*/evaluator/_*.py")
        if generic in path.read_text(encoding="utf-8")
    ]
    assert _EVALUATOR_MODULES
    assert not offenders


@pytest.mark.unittest
@pytest.mark.parametrize("module_name", _EVALUATOR_MODULES)
@pytest.mark.parametrize("model_output, cause, details", _CASES)
def test_invalid_output_message_is_actionable_and_privacy_safe(module_name, model_output, cause, details):
    """The message names the failure cause and safe metadata without echoing judge output."""
    module = importlib.import_module(f"...builtin.{module_name}", __package__)
    message = module._invalid_output_message(model_output)
    assert message.startswith("Evaluator returned invalid output: ")
    assert cause in message
    assert details in message
    assert _SECRET not in message
