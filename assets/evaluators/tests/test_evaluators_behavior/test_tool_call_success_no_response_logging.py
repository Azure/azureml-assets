# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests guarding against credential / PII leaks via response logging.

Several evaluator fallback paths previously logged the full raw payload
(``response``, ``query``, ``tool_definitions``, parsed ``tool_calls``) at
WARNING or DEBUG level. Those payloads are customer-controlled - they carry
tool arguments, tool results, user-pasted text, system prompts, file content,
database rows, and other data that can include credentials or PII. Logging
them - even at ``debug`` level - leaks that data into any sink that captures
the corresponding log level (CI logs, Azure ML run logs, App Insights, OTel
exporters, customer telemetry).

These tests assert that for each known fallback / debug path, the raw payload
never appears in captured log output at any level. They cover one file per
leak site; the synthetic ``LEAK_CANARY`` placeholder stands in for any sensitive
substring that might be present in real customer payloads.
"""

import logging

import pytest

from ...builtin.tool_call_success.evaluator._tool_call_success import (
    _reformat_tool_calls_results,
    _reformat_tool_definitions as _tcs_reformat_tool_definitions,
    _describe_response,
)
from ...builtin.tool_selection.evaluator._tool_selection import (
    reformat_conversation_history as _ts_reformat_conversation_history,
    _describe_payload as _ts_describe_payload,
)
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    reformat_conversation_history as _tia_reformat_conversation_history,
    _describe_payload as _tia_describe_payload,
)
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import (
    reformat_conversation_history as _tou_reformat_conversation_history,
    reformat_tool_definitions as _tou_reformat_tool_definitions,
    _describe_payload as _tou_describe_payload,
)


LEAK_CANARY = "leak-canary-xyzzy-do-not-log"


def _assert_secret_not_logged(caplog):
    leaks = [r for r in caplog.records if LEAK_CANARY in r.getMessage()]
    assert not leaks, (
        "Payload was written to logs - this leaks credentials / PII. "
        f"Offending records: {[(r.levelname, r.getMessage()) for r in leaks]}"
    )


# region tool_call_success._reformat_tool_calls_results


def test_tcs_exception_branch_does_not_log_response(caplog):
    """Parsing error in tool_call_success must not log the raw response."""

    class Boom:
        """Response that triggers a TypeError inside _get_tool_calls_results."""

        def __repr__(self):
            return f"<Response marker={LEAK_CANARY}>"

        def __iter__(self):
            raise TypeError("cannot iterate")

    logger = logging.getLogger("test_tcs_exception_branch_does_not_log_response")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = _reformat_tool_calls_results(Boom(), logger=logger)

    assert isinstance(result, Boom)
    _assert_secret_not_logged(caplog)


def test_tcs_empty_extraction_branch_does_not_log_response(caplog):
    """Empty-extraction fallback in tool_call_success must not log the raw response."""
    response = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"observed value {LEAK_CANARY}"}],
        }
    ]

    logger = logging.getLogger("test_tcs_empty_extraction_branch_does_not_log_response")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = _reformat_tool_calls_results(response, logger=logger)

    assert result is response
    _assert_secret_not_logged(caplog)


def test_tcs_tool_definitions_exception_does_not_log_payload(caplog):
    """Parsing error in tool_call_success._reformat_tool_definitions must not log the raw definitions."""

    class BoomDefs:
        def __repr__(self):
            return f"<ToolDefs marker={LEAK_CANARY}>"

        def __iter__(self):
            raise TypeError("cannot iterate")

    logger = logging.getLogger("test_tcs_tool_definitions_exception_does_not_log_payload")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = _tcs_reformat_tool_definitions(BoomDefs(), logger=logger)

    assert isinstance(result, BoomDefs)
    _assert_secret_not_logged(caplog)


# endregion


# region reformat_conversation_history (three siblings)


def _make_unparseable_query():
    """A query object whose iteration raises - reliably triggers the except branch
    of ``reformat_conversation_history`` across all three evaluator copies."""

    class BoomQuery:
        def __repr__(self):
            return f"<Query marker='{LEAK_CANARY}'>"

        def __iter__(self):
            raise TypeError("cannot iterate")

    return BoomQuery()


@pytest.mark.parametrize(
    "name,reformat",
    [
        ("tool_selection", _ts_reformat_conversation_history),
        ("tool_input_accuracy", _tia_reformat_conversation_history),
        ("tool_output_utilization", _tou_reformat_conversation_history),
    ],
)
def test_reformat_conversation_history_does_not_log_query(name, reformat, caplog):
    """All three reformat_conversation_history copies must not log the raw query."""
    query = _make_unparseable_query()
    logger = logging.getLogger(f"test_reformat_conversation_history_does_not_log_query[{name}]")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = reformat(query, logger=logger)

    assert result is query
    _assert_secret_not_logged(caplog)


# endregion


# region tool_output_utilization._reformat_tool_definitions


def test_tou_tool_definitions_exception_does_not_log_payload(caplog):
    """Parsing error in tool_output_utilization._reformat_tool_definitions must not log the raw definitions."""

    class BoomDefs:
        def __repr__(self):
            return f"<ToolDefs marker={LEAK_CANARY}>"

        def __iter__(self):
            raise TypeError("cannot iterate")

    logger = logging.getLogger("test_tou_tool_definitions_exception_does_not_log_payload")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        result = _tou_reformat_tool_definitions(BoomDefs(), logger=logger)

    assert isinstance(result, BoomDefs)
    _assert_secret_not_logged(caplog)


# endregion


# region _describe_payload / _describe_response adversarial robustness


# All four describe-helper copies must behave identically under hostile input:
# they must never raise, must never include the original value's repr, and
# must always produce a short, structural-only string. Parametrize over each
# helper so a regression in any one copy is caught.
DESCRIBE_HELPERS = [
    ("tool_call_success._describe_response", _describe_response),
    ("tool_selection._describe_payload", _ts_describe_payload),
    ("tool_input_accuracy._describe_payload", _tia_describe_payload),
    ("tool_output_utilization._describe_payload", _tou_describe_payload),
]


class _BadLen(list):
    def __len__(self):
        raise RuntimeError(f"len boom containing {LEAK_CANARY}")


class _BadDict(dict):
    def keys(self):
        raise RuntimeError(f"keys boom containing {LEAK_CANARY}")


class _BadObj:
    def __len__(self):
        raise ValueError(f"len boom containing {LEAK_CANARY}")

    def __repr__(self):
        return f"<BadObj marker={LEAK_CANARY}>"


def _adversarial_inputs():
    """Inputs that historically broke naive describe-by-repr implementations."""
    rec = {}
    rec["self"] = rec  # recursive dict - would blow up json.dumps / repr
    return [
        ("none", None),
        ("empty_list", []),
        ("empty_dict", {}),
        ("list_of_strings", [f"payload {LEAK_CANARY}", "x"]),
        ("list_of_ints", [1, 2, 3]),
        ("list_of_mixed", [{"role": "user"}, f"stray {LEAK_CANARY}", 42, None]),
        ("list_with_non_string_roles", [{"role": 1}, {"role": None}, {"role": ["weird"]}]),
        ("dict_with_int_keys", {1: f"value {LEAK_CANARY}", 2: "b"}),
        ("dict_with_mixed_keys", {"a": 1, 2: f"value {LEAK_CANARY}", (1, 2): "c"}),
        ("recursive_dict", rec),
        ("bytes_payload", f"prefix-{LEAK_CANARY}".encode("utf-8")),
        ("bytearray_payload", bytearray(LEAK_CANARY.encode("utf-8"))),
        ("set_payload", {f"a {LEAK_CANARY}", "b"}),
        ("tuple_payload", (f"a {LEAK_CANARY}", "b")),
        ("huge_list", [{"role": "user"}] * 10_000),
        ("bad_len_list", _BadLen([1, 2, 3])),
        ("bad_dict_keys", _BadDict({"a": LEAK_CANARY})),
        ("bad_obj_with_len_raising", _BadObj()),
    ]


@pytest.mark.parametrize("helper_name,helper", DESCRIBE_HELPERS)
@pytest.mark.parametrize("label,value", _adversarial_inputs())
def test_describe_helper_never_raises_and_never_leaks(helper_name, helper, label, value):
    """Every describe helper must (1) never raise and (2) never include the canary marker."""
    try:
        out = helper(value)
    except Exception as e:  # pragma: no cover - defensive
        pytest.fail(f"{helper_name} raised on input {label!r}: {type(e).__name__}: {e}")

    assert isinstance(out, str), f"{helper_name} returned non-str for {label!r}: {type(out)}"
    assert LEAK_CANARY not in out, (
        f"{helper_name} leaked the canary marker in output for input {label!r}: {out!r}"
    )


@pytest.mark.parametrize("helper_name,helper", DESCRIBE_HELPERS)
def test_describe_helper_truncates_long_role_lists(helper_name, helper):
    """A 10k-message conversation must not produce a 10k-long roles string."""
    huge = [{"role": "user"}] * 10_000
    out = helper(huge)
    # Cap chosen generously - real shape descriptions stay well under 256 chars.
    assert len(out) < 512, f"{helper_name} produced unbounded output ({len(out)} chars): {out[:200]}..."
    assert "len=10000" in out, f"{helper_name} did not report true length: {out}"


# endregion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
