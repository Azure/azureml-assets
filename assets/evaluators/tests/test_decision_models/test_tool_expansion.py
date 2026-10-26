# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise standalone tool evaluators through native transport without provider calls."""

import asyncio
import importlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.tool_call_success.evaluator._tool_call_success import ToolCallSuccessEvaluator
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import ToolInputAccuracyEvaluator
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import ToolOutputUtilizationEvaluator
from ...builtin.tool_selection.evaluator._tool_selection import ToolSelectionEvaluator
from ..test_evaluators_behavior import common_tool_test_data as data


EVALUATORS = [ToolCallSuccessEvaluator, ToolInputAccuracyEvaluator,
              ToolOutputUtilizationEvaluator, ToolSelectionEvaluator]
CONFIG = {"type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
          "azure_deployment": "Decision-1", "api_key": "test-only"}
CHAT_CONFIG = {"azure_endpoint": "https://example.openai.azure.com",
               "azure_deployment": "chat", "api_key": "test-only"}
PROMPTY_LOAD = "azure.ai.evaluation._evaluators._common._model_runners.AsyncPrompty.load"
# The inputs each chat prompty declares and renders; the native request must carry exactly these.
RENDERED_INPUTS = {
    ToolCallSuccessEvaluator: ("tool_calls", "tool_definitions"),
    ToolInputAccuracyEvaluator: ("query", "tool_calls", "tool_definitions"),
    ToolOutputUtilizationEvaluator: ("query", "response", "tool_definitions"),
    ToolSelectionEvaluator: ("query", "tool_calls", "tool_definitions"),
}
SERIALIZED = {"query": "What is my horoscope? I am an Aquarius.",
              "response": '[TOOL_CALL] get_horoscope(sign="Aquarius") [TOOL_RESULT] "Meet an otter"',
              "tool_definitions": "get_horoscope(sign: string): returns a horoscope."}


def inputs(shape="structured"):
    """Provide a completed turn whose call and result have the same identifier."""
    result = {"query": deepcopy(data.LOCAL_CALLS_QUERY), "response": deepcopy(data.LOCAL_CALLS_RESPONSE),
              "tool_definitions": deepcopy(data.LOCAL_CALLS_TOOL_DEFINITIONS)}
    if shape == "messages":
        result = {"messages": result["query"] + result["response"],
                  "tool_definitions": result["tool_definitions"]}
    elif shape == "serialized":
        result = dict(SERIALIZED)
    return result


def expected_state(evaluator_type, shape):
    """Build the exact evidence each prompty renders for the fixture shape, which the native request must equal."""
    if shape == "serialized":
        prepared = {"query": SERIALIZED["query"], "response": SERIALIZED["response"],
                    "tool_calls": SERIALIZED["response"], "tool_definitions": SERIALIZED["tool_definitions"]}
    else:
        tool_calls = {ToolCallSuccessEvaluator: data.LOCAL_CALLS_TCS_EXPECTED_FLOW_TOOL_CALLS,
                      ToolInputAccuracyEvaluator: data.LOCAL_CALLS_TIA_EXPECTED_FLOW_TOOL_CALLS,
                      ToolSelectionEvaluator: data.LOCAL_CALLS_TS_EXPECTED_FLOW_TOOL_CALLS}
        formatted_definitions = evaluator_type in (ToolCallSuccessEvaluator, ToolOutputUtilizationEvaluator)
        prepared = {"query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
                    "response": data.LOCAL_CALLS_TOU_EXPECTED_FLOW_RESPONSE,
                    "tool_calls": tool_calls.get(evaluator_type),
                    "tool_definitions": (data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_DEFINITIONS_STR
                                         if formatted_definitions else data.LOCAL_CALLS_TOOL_DEFINITIONS)}
    return {name: prepared[name] for name in RENDERED_INPUTS[evaluator_type]}


def native(key, probability=0.9, status=1):
    """Build one paid response containing both applicability and score answers."""
    return {"model": "actual-model", "usage": {"input_tokens": 30, "output_tokens": 4},
            "answers": {"status": {"type": "noul", "noul": status},
                        key: {"type": "noul", "noul": probability}}}


def native_evaluator(evaluator_type, probability=0.9, status=1, **kwargs):
    """Construct the real evaluator and mock only its network boundary."""
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Native construction must not load a chat flow")):
        evaluator = evaluator_type(CONFIG, **kwargs)
    evaluator._model_runner._post = AsyncMock(return_value=native(evaluator._RESULT_KEY, probability, status))
    return evaluator


def request(evaluator):
    """Inspect the SDK's actual serialized request."""
    content, = evaluator._model_runner._post.call_args.args
    body = json.loads(content)
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == CONFIG["azure_deployment"]
    return body


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
@pytest.mark.parametrize("shape", ["structured", "serialized", "messages"])
def test_native_and_chat_share_prepared_input_and_binary_results(evaluator_type, shape):
    """Keep preprocessing, binary scoring, and input ownership identical across runners."""
    payload = inputs(shape)
    original = deepcopy(payload)
    evaluator = native_evaluator(evaluator_type)
    native_result = evaluator(**payload)
    evaluator._model_runner._post.assert_awaited_once()
    body = request(evaluator)
    key = evaluator._RESULT_KEY
    assert set(body["questions"]) == {"status", key}
    assert all(question["type"] == "noul" for question in body["questions"].values())

    chat = evaluator_type(CHAT_CONFIG)
    flow = chat._flow
    rendered = []

    async def render_only(timeout, **prepared):
        rendered.append(json.dumps(flow.render(**prepared)))
        return {"llm_output": {"score": 1, "status": "completed", "reason": "Mocked judgment."}}

    chat._flow = AsyncMock(side_effect=render_only)
    chat_result = chat(**payload)
    chat._flow.assert_awaited_once()
    chat_input = dict(chat._flow.call_args.kwargs)
    chat_input.pop("timeout")
    rendered_names = RENDERED_INPUTS[evaluator_type]
    assert set(rendered_names) == set(flow._inputs)
    assert body["state"] == expected_state(evaluator_type, shape)
    assert body["state"] == {name: chat_input[name] for name in rendered_names}
    assert None not in body["state"].values()
    assert rendered and "get_horoscope" in rendered[0]
    assert payload == original
    for suffix in ["", "_score", "_passed", "_result", "_status", "_threshold"]:
        assert native_result[key + suffix] == chat_result[key + suffix]
    assert native_result[key] == 1
    properties = native_result[key + "_properties"]
    assert properties["total_tokens"] == 34
    assert properties["model"] == "actual-model"
    assert native_result[key + "_total_tokens"] == 34
    assert properties["decision_model"]["probability_threshold"] == 0.75
    assert properties["decision_model"]["status_probability_threshold"] == 0.5
    assert "No model rationale" in native_result[key + "_reason"]
    # The native API provides no parameter counts, failed tool names, or selection counts.
    assert not {"parameter_extraction_accuracy", "tool_selection_accuracy", "failed_tools"}.intersection(properties)


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
def test_paid_native_skip_retains_usage_and_does_not_parse_score(evaluator_type):
    """An unusable score is ignored when the separate applicability question skips."""
    evaluator = native_evaluator(evaluator_type, probability="unusable", status=0.49)
    key = evaluator._RESULT_KEY
    result = asyncio.run(evaluator._to_async()(**inputs()))
    evaluator._model_runner._post.assert_awaited_once()
    assert result[key] is None and result[key + "_passed"] is None
    assert result[key + "_status"] == "skipped"
    assert result[key + "_result"] == "not_applicable"
    assert result[key + "_total_tokens"] == 34
    assert result[key + "_properties"]["total_tokens"] == 34
    assert result[key + "_properties"]["model"] == "actual-model"


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
@pytest.mark.parametrize("probability, expected", [(0.799, 0), (0.8, 1)])
def test_probability_override_keeps_applicability_independent(evaluator_type, probability, expected):
    """Use the configured score cutoff but the fixed applicability boundary of 0.5."""
    evaluator = native_evaluator(evaluator_type, probability=probability, status=0.5, probability_threshold=0.8)
    result = evaluator(**inputs())
    key = evaluator._RESULT_KEY
    assert result[key] == expected
    assert result[key + "_threshold"] == 1
    assert result[key + "_status"] == "completed"
    assert "probability_threshold" not in CONFIG


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
@pytest.mark.parametrize("cutoff", [0, 1])
def test_probability_cutoff_accepts_schema_boundaries(evaluator_type, cutoff):
    """Both endpoints of the optional public probability range are valid."""
    evaluator = native_evaluator(evaluator_type, probability=cutoff, probability_threshold=cutoff)
    result = evaluator(**inputs())
    assert result[evaluator._RESULT_KEY] == 1


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
@pytest.mark.parametrize("cutoff", [-0.01, 1.01, True, "0.75"])
def test_probability_cutoff_rejects_invalid_values(evaluator_type, cutoff):
    """Only numeric cutoffs in the schema range are accepted before any request."""
    with patch(PROMPTY_LOAD) as load:
        with pytest.raises(EvaluationException):
            evaluator_type(CONFIG, probability_threshold=cutoff)
        load.assert_not_called()


@pytest.mark.parametrize("evaluator_type", [ToolCallSuccessEvaluator, ToolSelectionEvaluator])
def test_binary_pass_semantics_preserve_nondefault_reported_threshold(evaluator_type):
    """These standalone binary contracts still fail score zero with threshold zero."""
    evaluator = native_evaluator(evaluator_type, probability=0.2, threshold=0)
    key = evaluator._RESULT_KEY
    result = evaluator(**inputs())
    assert result[key] == 0 and result[key + "_passed"] is False
    assert result[key + "_threshold"] == 0


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
def test_intermediate_response_is_a_free_skip(evaluator_type):
    """Existing validation skips unfinished turns without a native request."""
    evaluator = native_evaluator(evaluator_type)
    payload = inputs()
    payload["response"] = payload["response"][:1]
    result = evaluator(**payload)
    key = evaluator._RESULT_KEY
    evaluator._model_runner._post.assert_not_called()
    assert result[key] is None
    assert result[key + "_properties"] is None
    assert result[key + "_total_tokens"] == 0


@pytest.mark.parametrize("evaluator_type", [ToolInputAccuracyEvaluator, ToolSelectionEvaluator])
def test_no_calls_is_a_free_skip_for_call_extracting_evaluators(evaluator_type):
    """Retain their pre-request no-call checks instead of inventing a score."""
    evaluator = native_evaluator(evaluator_type)
    payload = inputs()
    payload["response"] = [{"role": "assistant", "content": [{"type": "text", "text": "Hello."}]}]
    result = evaluator(**payload)
    evaluator._model_runner._post.assert_not_called()
    assert result[evaluator._RESULT_KEY + "_status"] == "skipped"


@pytest.mark.parametrize("status", ["failed", "incomplete"])
@pytest.mark.parametrize("location", ["call", "result"])
def test_success_runtime_failure_never_calls_model(status, location):
    """Explicit call or matched-result runtime failure remains an authoritative failure."""
    evaluator = native_evaluator(ToolCallSuccessEvaluator)
    payload = inputs()
    payload["response"][0 if location == "call" else 1]["content"][0]["status"] = status
    result = evaluator(**payload)
    evaluator._model_runner._post.assert_not_called()
    assert result["tool_call_success"] == 0
    assert result["tool_call_success_status"] == "completed"
    assert result["tool_call_success_total_tokens"] == 0
    assert result["tool_call_success_properties"]["failed_tools"] == "get_horoscope"


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
def test_paid_chat_skip_retains_known_usage(evaluator_type):
    """A paid chat skip retains metadata just like a paid native skip."""
    evaluator = evaluator_type(CHAT_CONFIG)
    evaluator._flow = AsyncMock(return_value={
        "llm_output": {"score": None, "status": "skipped", "reason": "Not enough evidence.",
                       "properties": {"source": "judge"}},
        "input_token_count": 13, "output_token_count": 3, "total_token_count": 16,
        "model_id": "actual-chat", "finish_reason": "stop", "sample_output": "Known output",
    })
    key = evaluator._RESULT_KEY
    result = evaluator(**inputs())
    assert result[key] is None and result[key + "_passed"] is None
    assert result[key + "_result"] == "not_applicable"
    assert result[key + "_total_tokens"] == 16
    assert result[key + "_properties"]["total_tokens"] == 16
    assert result[key + "_properties"]["source"] == "judge"
    assert result[key + "_model"] == "actual-chat"
    assert result[key + "_finish_reason"] == "stop"
    assert result[key + "_sample_output"] == "Known output"


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
@pytest.mark.parametrize("bad_answer", [
    {"type": "noul", "noul": 1.1}, {"type": "noul", "noul": True}, {"type": "noul", "noul": "0.9"},
])
def test_bad_native_answer_retains_error_usage(evaluator_type, bad_answer):
    """Invalid paid score answers fail clearly and retain known request usage."""
    evaluator = native_evaluator(evaluator_type)
    key = evaluator._RESULT_KEY
    evaluator._model_runner._post.return_value["answers"][key] = bad_answer
    with pytest.raises(EvaluationException) as raised:
        evaluator(**inputs())
    assert raised.value.properties["total_tokens"] == 34
    assert raised.value.properties["model"] == "actual-model"


@pytest.mark.parametrize("evaluator_type", EVALUATORS)
def test_old_sdk_capability_and_invalid_config_fail_before_chat_load(evaluator_type):
    """Older SDKs retain chat support but reject opt-in native configuration explicitly."""
    module = importlib.import_module(evaluator_type.__module__)
    with patch.object(module, "_SDK_SUPPORTS_DECISION", False), patch(PROMPTY_LOAD) as load:
        with pytest.raises(EvaluationException, match="version with decision model support"):
            evaluator_type(CONFIG)
        load.assert_not_called()
    with patch(PROMPTY_LOAD) as load:
        with pytest.raises(EvaluationException):
            evaluator_type({**CONFIG, "azure_endpoint": "not-a-url"})
        with pytest.raises(EvaluationException, match="only supported for Decision"):
            evaluator_type(CHAT_CONFIG, probability_threshold=0.8)
        load.assert_not_called()


@pytest.mark.parametrize("evaluator_type", [ToolInputAccuracyEvaluator, ToolOutputUtilizationEvaluator])
def test_optional_definitions_skip_before_native_request(evaluator_type):
    """Missing optional tool definitions skip the judgment without a native request."""
    evaluator = native_evaluator(evaluator_type)
    payload = inputs()
    payload.pop("tool_definitions")
    result = evaluator(**payload)
    assert result[evaluator._RESULT_KEY + "_status"] == "skipped"
    evaluator._model_runner._post.assert_not_called()


def test_success_optional_definitions_and_matched_execution_evidence():
    """Success keeps matched results and supports definitions being omitted."""
    evaluator = native_evaluator(ToolCallSuccessEvaluator)
    payload = inputs()
    payload.pop("tool_definitions")
    payload["response"].insert(1, {"role": "tool", "tool_call_id": "unmatched-id", "content": [
        {"type": "function_call_output", "function_call_output": "UNMATCHED_RESULT"}]})
    evaluator(**payload)
    state = request(evaluator)["state"]
    assert set(state) == {"tool_calls"}
    assert "get_horoscope" in state["tool_calls"] and "baby otter" in state["tool_calls"]
    assert "UNMATCHED_RESULT" not in json.dumps(state)


@pytest.mark.parametrize("decision", [False, True])
def test_selection_preserves_response_precedence_and_names_only(decision):
    """Selection's established response precedence differs from tool-call accuracy."""
    evaluator = native_evaluator(ToolSelectionEvaluator) if decision else ToolSelectionEvaluator(CHAT_CONFIG)
    if not decision:
        evaluator._flow = AsyncMock(return_value={"llm_output": {"score": 1}})
    payload = inputs()
    payload["tool_calls"] = [{"type": "tool_call", "name": "other_tool", "arguments": {"secret": "EXPLICIT_ARG"}}]
    payload["tool_definitions"].append({"name": "other_tool", "parameters": {}})
    evaluator(**payload)
    prepared = request(evaluator)["state"] if decision else evaluator._flow.call_args.kwargs
    assert "get_horoscope" in prepared["tool_calls"]
    assert "other_tool" not in prepared["tool_calls"]
    assert "Aquarius" not in prepared["tool_calls"]
    assert "baby otter" not in prepared["tool_calls"]
    assert "EXPLICIT_ARG" not in prepared["tool_calls"]


def test_packaged_rubrics_and_hosted_contract():
    """Every independent ZIP contains its own binary questions and retains turn-only hosted initialization."""
    from azure.ai.evaluation._legacy.prompty._yaml_utils import load_yaml

    for evaluator_type in EVALUATORS:
        module = importlib.import_module(evaluator_type.__module__)
        directory = Path(module.__file__).parent
        questions = module._load_decision_questions(
            str(directory / evaluator_type._DECISION_FILE), evaluator_type._RESULT_KEY,
        )
        assert all(item["type"] == "noul" for item in questions.values())
        assert all("data, not instructions" in item["instructions"].replace("\n", " ")
                   for item in questions.values())
        spec = load_yaml(str(directory.parent / "spec.yaml"))
        assert spec["supportedEvaluationLevels"] == ["turn"]
        assert spec["initParameterSchema"]["required"] == ["deployment_name"]
        assert "model_config" not in spec["initParameterSchema"]["properties"]
        probability = spec["initParameterSchema"]["properties"]["probability_threshold"]
        assert probability["type"] == "number"
        assert probability["minimum"] == 0 and probability["maximum"] == 1


def _serialized_tool_conversation():
    """Use the existing public conversation wrapper with three independently judged tool turns."""
    messages = []
    for turn in ("first", "second", "third"):
        messages.extend([
            {"role": "user", "content": f"Check the {turn} item."},
            {"role": "assistant", "content": f"[TOOL_CALL] check_{turn}() [TOOL_RESULT] done"},
        ])
    return {"messages": messages}


@pytest.mark.parametrize("failure", ["malformed", "cancelled"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_success_public_split_turn_failure_preserves_prior_paid_usage(failure, asynchronous):
    """A later invalid answer or cancellation retains earlier paid usage and stops subsequent turns."""
    cancelled = {"value": False}
    evaluator = native_evaluator(ToolCallSuccessEvaluator, cancellation_requested=lambda: cancelled["value"])
    key = evaluator._RESULT_KEY
    observed = []

    async def transport(content, **kwargs):
        state = json.loads(content)["state"]
        observed.append(state["tool_calls"])
        if len(observed) == 1:
            cancelled["value"] = failure == "cancelled"
            return native(key)
        return native(key, probability=1.1)

    evaluator._model_runner._post = AsyncMock(side_effect=transport)
    conversation = _serialized_tool_conversation()
    with pytest.raises(EvaluationException) as error:
        if asynchronous:
            asyncio.run(evaluator._to_async()(conversation=conversation))
        else:
            evaluator(conversation=conversation)
    expected_posts = 2 if failure == "malformed" else 1
    assert evaluator._model_runner._post.await_count == expected_posts
    assert "check_first" in observed[0]
    assert not any("check_third" in state for state in observed)
    assert error.value.properties["prompt_tokens"] == 30 * expected_posts
    assert error.value.properties["completion_tokens"] == 4 * expected_posts
    assert error.value.properties["total_tokens"] == 34 * expected_posts
    assert error.value.properties["model"] == "actual-model"
    details = error.value.properties["decision_model"]
    assert details["result_usage_scope"] == "reported_turns_only"
    assert "execution" not in details
    assert ("cancelled" in str(error.value)) is (failure == "cancelled")


@pytest.mark.parametrize("with_definitions", [False, True])
def test_utilization_public_conversation_rejection_does_not_invent_split_support(with_definitions):
    """Reject the unsupported conversation wrapper independently of optional definitions."""
    evaluator = native_evaluator(ToolOutputUtilizationEvaluator)
    kwargs = {"conversation": _serialized_tool_conversation()}
    if with_definitions:
        kwargs["tool_definitions"] = inputs()["tool_definitions"]
    with pytest.raises(EvaluationException):
        evaluator(**kwargs)
    evaluator._model_runner._post.assert_not_called()
