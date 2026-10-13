# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Composite contracts exercised against the actual shared Decision SDK."""

from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import AsyncMock

import jsonschema
import pandas as pd
import pytest
import yaml

from azure.ai.evaluation._evaluate._evaluate import (
    _calculate_aoai_evaluation_summary,
    _convert_single_row_to_aoai_format,
    _extract_from_evaluator_base,
    _extract_metrics_from_evaluator_name,
)
from azure.ai.evaluation._exceptions import EvaluationException
from azure.ai.evaluation._evaluate._batch_run._run_submitter_client import RunSubmitterClient
from ...builtin.output_quality.evaluator._output_quality import OutputQualityEvaluator
from ...builtin.tool_use_quality.evaluator._tool_use_quality import ToolUseQualityEvaluator


CONFIG = {
    "type": "decision",
    "azure_endpoint": "https://example.services.ai.azure.com",
    "azure_deployment": "Decision-1",
    "api_key": "test-only",
}
CHAT_CONFIG = {"azure_endpoint": CONFIG["azure_endpoint"], "azure_deployment": "chat", "api_key": "test-only"}
COMPOSITES = [OutputQualityEvaluator, ToolUseQualityEvaluator]
INPUT = {"query": "Say hello", "response": "Hello", "tool_definitions": [{"name": "greet", "parameters": {}}]}


def _native(evaluator_class, skipped=False):
    answers = {}
    for member in evaluator_class._EVALUATORS:
        name = member["name"]
        answers[f"{name}_status"] = {"type": "noul", "noul": 0 if skipped else 1}
        answers[name] = (
            {"type": "noul", "noul": 0.9} if member["max"] == 1 else
            {"type": "score", "score": 4, "confidence": 1,
             "probabilities": {str(index): float(index == 4) for index in range(5)}}
        )
    return {"model": "actual-model", "answers": answers, "usage": {"input_tokens": 30, "output_tokens": 4}}


def _metadata(evaluator, source):
    metric = evaluator._RESULT_KEY
    expected = [metric, *(member["name"] for member in evaluator._EVALUATORS)]
    if source == "sdk_instance":
        criteria_type, name, metrics, _ = _extract_from_evaluator_base(evaluator, "composite")
        assert criteria_type == "azure_ai_evaluator" and name == metric
        assert metrics == expected
    elif source == "sdk_builtin":
        name = f"builtin.{metric}"
        metrics = _extract_metrics_from_evaluator_name(
            name, "composite", "azure_ai_evaluator", logging.getLogger(__name__), "test", "run",
        )
        assert metrics == expected
    else:
        name = f"builtin.{metric}"
        metrics = expected
    return {"composite": {"type": "azure_ai_evaluator", "evaluator_name": name, "metrics": metrics}}


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("threshold", [
    3, [], {"unknown": 1}, {"tool_definitions": 1}, {"model_config": 1},
])
def test_threshold_mapping_rejects_invalid_shape(evaluator_class, decision, threshold):
    with pytest.raises(EvaluationException, match="threshold must be a mapping"):
        evaluator_class(CONFIG if decision else CHAT_CONFIG, threshold=threshold)


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("value", [True, False, "3", None, float("nan"), float("inf"), -float("inf"), 10**400])
def test_threshold_mapping_rejects_invalid_values(evaluator_class, decision, value):
    for member in evaluator_class._EVALUATORS:
        with pytest.raises(EvaluationException, match=f"threshold for {member['name']} must be a number"):
            evaluator_class(CONFIG if decision else CHAT_CONFIG, threshold={member["name"]: value})


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("position", ["below", "minimum", "fractional", "maximum", "above"])
def test_threshold_member_bounds_and_copy(evaluator_class, decision, position):
    config = deepcopy(CONFIG if decision else CHAT_CONFIG)
    original = deepcopy(config)
    for member in evaluator_class._EVALUATORS:
        value = {
            "below": member["min"] - 0.1, "minimum": member["min"],
            "fractional": member["min"] + (member["max"] - member["min"]) * 0.625,
            "maximum": member["max"], "above": member["max"] + 0.1,
        }[position]
        thresholds = {member["name"]: value}
        if position in ("below", "above"):
            with pytest.raises(EvaluationException, match=f"threshold for {member['name']}"):
                evaluator_class(config, threshold=thresholds)
        else:
            evaluator = evaluator_class(config, threshold=thresholds)
            assert evaluator._threshold_for(member["name"]) == value
            thresholds[member["name"]] = -100
            assert evaluator._threshold_for(member["name"]) == value
        assert config == original
    defaults = evaluator_class(config, threshold={})
    assert defaults._threshold == {member["name"]: member["default_threshold"] for member in defaults._EVALUATORS}


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
def test_member_threshold_schema_matches_constructor(evaluator_class):
    module = __import__(evaluator_class.__module__, fromlist=["__file__"])
    spec = yaml.safe_load((Path(module.__file__).parent.parent / "spec.yaml").read_text(encoding="utf-8"))
    schema = spec["initParameterSchema"]
    for binding in ({"deployment_name": "resolved-deployment"}, {"model_config": {
        key: value for key, value in CONFIG.items() if key != "api_key"
    }}):
        jsonschema.validate({**binding, "threshold": {}, "probability_threshold": 0.8}, schema)
        for member in evaluator_class._EVALUATORS:
            for value in (member["min"], member["min"] + (member["max"] - member["min"]) * 0.625, member["max"]):
                jsonschema.validate({**binding, "threshold": {member["name"]: value}}, schema)
            for value in (True, "3", None, member["min"] - 0.1, member["max"] + 0.1):
                with pytest.raises(jsonschema.ValidationError):
                    jsonschema.validate({**binding, "threshold": {member["name"]: value}}, schema)
        for invalid in (3, [], {"unknown": 1}):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({**binding, "threshold": invalid}, schema)


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("skipped", [False, True])
@pytest.mark.parametrize("metadata_source", ["catalog", "sdk_instance", "sdk_builtin"])
def test_public_converter_and_rollup_count_shared_request_once(evaluator_class, skipped, metadata_source):
    evaluator = evaluator_class(CONFIG)
    post = AsyncMock(return_value=_native(evaluator_class, skipped))
    evaluator._decision_model_runner._post = post
    output = evaluator(**deepcopy(INPUT))
    post.assert_awaited_once()
    metric = evaluator._RESULT_KEY
    row = {f"outputs.composite.{key}": value for key, value in output.items()}
    logger = logging.getLogger(__name__)
    metadata = _metadata(evaluator, metadata_source)
    converted = _convert_single_row_to_aoai_format(row, 0, "test", "run", 0, metadata, None, logger)
    with_usage = [result for result in converted["results"] if result.get("sample", {}).get("usage")]
    assert len(with_usage) == 1
    assert with_usage[0]["metric"] == metric
    assert with_usage[0]["sample"]["model"] == "actual-model"
    assert with_usage[0]["sample"]["usage"] == {
        "prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34,
    }
    assert {result["metric"] for result in converted["results"]} >= {
        member["name"] for member in evaluator._EVALUATORS
    }
    assert all(result["name"] == "composite" for result in converted["results"])
    assert all(result["status"] == ("skipped" if skipped else "completed") for result in converted["results"])
    summary = _calculate_aoai_evaluation_summary([converted], logger, metadata)
    assert summary["per_model_usage"] == [{
        "model_name": "actual-model", "invocation_count": 1,
        "prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34, "cached_tokens": 0,
    }]


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.parametrize("level", [None, "turn", "conversation"])
@pytest.mark.parametrize("definitions", [None, [], INPUT["tool_definitions"]])
def test_public_data_schema_matches_native_input_routes(evaluator_class, conversation, level, definitions):
    module = __import__(evaluator_class.__module__, fromlist=["__file__"])
    schema = yaml.safe_load((Path(module.__file__).parent.parent / "spec.yaml").read_text(
        encoding="utf-8"))["dataMappingSchema"]
    inputs = {"messages": [
        {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello"},
    ]} if conversation else {"response": "Hello"}
    if not conversation and evaluator_class is ToolUseQualityEvaluator:
        inputs["query"] = "Say hello"
    if definitions is not None:
        inputs["tool_definitions"] = deepcopy(definitions)
    jsonschema.validate(inputs, schema)
    evaluator = evaluator_class(CONFIG, evaluation_level=level)
    evaluator._decision_model_runner._post = AsyncMock(return_value=_native(evaluator_class))
    output = evaluator(**deepcopy(inputs))
    evaluator._decision_model_runner._post.assert_awaited_once()
    assert output[evaluator._RESULT_KEY] == 1
    for invalid in ({}, {"query": "Say hello"}, {"messages": []}, {"messages": "bad"},
                    {**inputs, "query": 12}, {**inputs, "response": False},
                    {**inputs, "tool_definitions": [12]}, {**inputs, "tool_definitions": None}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("query_list", [False, True])
@pytest.mark.parametrize("response_list", [False, True])
def test_mixed_input_shapes_preserve_conversation_evidence(evaluator_class, query_list, response_list):
    query = [{"role": "user", "content": "Say hello"}] if query_list else "Say hello"
    response = [{"role": "assistant", "content": "Hello"}] if response_list else "Hello"
    evaluator = evaluator_class(CONFIG, evaluation_level="conversation")
    post = AsyncMock(return_value=_native(evaluator_class))
    evaluator._decision_model_runner._post = post
    evaluator(query=deepcopy(query), response=deepcopy(response))
    post.assert_awaited_once()
    state = json.loads(post.call_args.args[0])["state"]
    assert "User turn 1:" in state["messages"] and "Say hello" in state["messages"]
    assert "Agent turn 1:" in state["messages"] and "Hello" in state["messages"]
    assert state["messages"].count("User turn ") == 1
    assert state["messages"].count("Agent turn ") == 1


def test_binary_public_member_threshold_is_independent_of_probability_cutoff():
    evaluator = OutputQualityEvaluator(CONFIG, threshold={"task_completion": 0})
    payload = _native(OutputQualityEvaluator)
    payload["answers"]["task_completion"]["noul"] = 0.1
    evaluator._decision_model_runner._post = AsyncMock(return_value=payload)
    output = evaluator(**INPUT)
    assert output["task_completion_score"] == 0
    assert output["task_completion_threshold"] == 0
    assert output["task_completion_result"] == "pass"
    member = output["output_quality_evaluators"]["task_completion"]
    assert member["passed"] is True and member["result"] == "pass"
    assert member["properties"]["decision_model"]["probability_threshold"] == 0.75


@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("conversation", [False, True])
def test_supplied_invalid_tool_definitions_are_not_ignored(decision, conversation):
    evaluator = ToolUseQualityEvaluator(CONFIG if decision else CHAT_CONFIG)
    inputs = {"messages": [
        {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello"},
    ]} if conversation else {"query": "Say hello", "response": "Hello"}
    with pytest.raises(EvaluationException, match="name"):
        evaluator(**inputs, tool_definitions=[{"parameters": {}}])


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
def test_plain_execution_controls_and_credential_reach_shared_runner(evaluator_class):
    credential = object()

    def cancelled():
        return False

    controls = {
        "timeout": 17, "timeout_seconds": 3, "max_retries": 0,
        "cancellation_requested": cancelled, "max_request_bytes": 40000,
        "max_state_question_bytes": 30000, "max_response_bytes": 10000,
    }
    config = {key: value for key, value in CONFIG.items() if key != "api_key"}
    original = deepcopy(config)
    evaluator = evaluator_class(config, credential=credential, **controls)
    assert config == original
    runner = evaluator._decision_model_runner
    assert runner._credential is credential
    assert evaluator._LLM_CALL_TIMEOUT == 17
    for name, value in controls.items():
        if name != "timeout":
            assert getattr(runner, f"_{name}") == value
    for strategies in (evaluator._decision_strategies, evaluator._multi_turn_decision_strategies):
        assert all(strategy.runner is runner and strategy.timeout == 17 for strategy in strategies.values())
    for name, value in controls.items():
        with pytest.raises(EvaluationException, match="only supported"):
            evaluator_class(CHAT_CONFIG, **{name: value})


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("field", ["temperature", "top_p"])
def test_decision_configuration_rejects_chat_model_parameters(evaluator_class, field):
    with pytest.raises(EvaluationException, match="Unsupported decision model configuration"):
        evaluator_class({**CONFIG, field: 0.5})


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("decision", [False, True])
@pytest.mark.parametrize("inputs", [
    {}, {"query": "Say hello"}, {"messages": []}, {"messages": "invalid"},
    {"query": "Say hello", "response": False},
])
def test_invalid_public_input_is_rejected_before_model_execution(evaluator_class, decision, inputs):
    evaluator = evaluator_class(CONFIG if decision else CHAT_CONFIG)
    if decision:
        evaluator._decision_model_runner._post = AsyncMock(side_effect=AssertionError("Unexpected model request"))
    else:
        evaluator._flow = AsyncMock(side_effect=AssertionError("Unexpected model request"))
        evaluator._multi_turn_flow = AsyncMock(side_effect=AssertionError("Unexpected model request"))
    with pytest.raises(EvaluationException):
        evaluator(**deepcopy(inputs))


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("level", ["turn", "conversation"])
@pytest.mark.parametrize("include_query", [False, True])
@pytest.mark.parametrize("include_definitions", [False, True, None])
def test_actual_prompty_render_receives_optional_definitions(
    evaluator_class, level, include_query, include_definitions,
):
    evaluator = evaluator_class(CHAT_CONFIG, evaluation_level=level)
    actual_turn = evaluator._flow
    actual_conversation = evaluator._multi_turn_flow
    captured = []
    llm_output = {"llm_output": {
        member["name"]: {"score": member["max"], "status": "completed", "reason": "test"}
        for member in evaluator._EVALUATORS
    }}

    async def render_only(timeout, **prepared):
        flow = actual_conversation if "messages" in prepared else actual_turn
        captured.append((prepared, flow.render(**prepared)))
        return llm_output

    evaluator._flow = AsyncMock(side_effect=render_only)
    evaluator._multi_turn_flow = AsyncMock(side_effect=render_only)
    inputs = {"response": "Hello"}
    if include_query:
        inputs["query"] = "Say hello"
    if include_definitions:
        inputs["tool_definitions"] = deepcopy(INPUT["tool_definitions"])
    elif include_definitions is None:
        inputs["tool_definitions"] = None
    if not include_query and evaluator_class is ToolUseQualityEvaluator:
        with pytest.raises(EvaluationException, match="[Qq]uery"):
            evaluator(**inputs)
        assert captured == []
        return
    assert evaluator(**inputs)[evaluator._RESULT_KEY] == 1
    assert len(captured) == 1
    prepared, messages = captured[0]
    assert "Hello" in json.dumps(messages)
    if include_query:
        assert "Say hello" in json.dumps(messages)
    if include_definitions:
        assert "greet" in json.dumps(prepared["tool_definitions"])
        assert "greet" in json.dumps(messages)
    elif level == "turn":
        assert prepared["tool_definitions"] == []


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("level", ["turn", "conversation"])
def test_python_optional_definition_none_follows_shared_sdk_omission(evaluator_class, level):
    evaluator = evaluator_class(CONFIG, evaluation_level=level)
    post = AsyncMock(return_value=_native(evaluator_class))
    evaluator._decision_model_runner._post = post
    inputs = {"query": "Say hello", "response": "Hello"}
    missing = evaluator(**inputs)
    explicit_none = evaluator(**inputs, tool_definitions=None)
    assert missing == explicit_none
    assert post.await_count == 2
    assert post.await_args_list[0].args[0] == post.await_args_list[1].args[0]


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("root_last", [False, True])
@pytest.mark.parametrize("metadata_source", ["catalog", "sdk_instance", "sdk_builtin"])
def test_actual_batch_failure_preserves_known_usage_once(evaluator_class, root_last, metadata_source):
    evaluator = evaluator_class(CONFIG)
    payload = _native(evaluator_class)
    payload["answers"][f"{evaluator._EVALUATORS[0]['name']}_status"]["noul"] = 2
    post = AsyncMock(return_value=payload)
    evaluator._decision_model_runner._post = post
    client = RunSubmitterClient()
    try:
        run = client.run(
            evaluator, pd.DataFrame([deepcopy(INPUT)]), evaluator_name="composite",
            column_mapping={key: f"${{data.{key}}}" for key in INPUT},
        )
        summary = client.get_run_summary(run)
    finally:
        client._thread_pool.shutdown(wait=True)
    post.assert_awaited_once()
    assert summary["failed_lines"] == 1
    error_metadata = summary["per_line_error_metadata"][0]
    metric = evaluator._RESULT_KEY
    members = [member["name"] for member in evaluator._EVALUATORS]
    metadata = _metadata(evaluator, metadata_source)
    if root_last:
        metadata["composite"]["metrics"] = [*members, metric]
    logger = logging.getLogger(__name__)
    converted = _convert_single_row_to_aoai_format(
        {}, 0, "test", "run", 0, metadata, {"composite": summary}, logger,
    )
    assert len(converted["results"]) == len(members) + 1
    for result in converted["results"]:
        assert result["name"] == "composite"
        assert result["score"] is None and result["passed"] is None
        assert result["status"] == "error"
        assert result["properties"]["decision_model"] == error_metadata["properties"]["decision_model"]
        assert result["sample"]["error"]["code"] == error_metadata["code"]
        assert result["sample"]["error"]["blame"] == error_metadata["blame"]
    with_usage = [result for result in converted["results"] if result["sample"].get("usage")]
    assert len(with_usage) == 1
    assert with_usage[0]["metric"] == metric
    rollup = _calculate_aoai_evaluation_summary([converted], logger, metadata)
    assert rollup["per_model_usage"] == [{
        "model_name": "actual-model", "invocation_count": 1,
        "prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34, "cached_tokens": 0,
    }]


@pytest.mark.parametrize("evaluator_class", COMPOSITES)
@pytest.mark.parametrize("version", ["1.17.0", "1.18.2"])
def test_actual_released_wheel_composite_package(tmp_path, evaluator_class, version):
    target = Path(tempfile.gettempdir()) / f"azureml-compat-sdk-{version}"
    assert (target / "azure" / "ai" / "evaluation" / "__init__.py").is_file(), target
    module = __import__(evaluator_class.__module__, fromlist=["__file__"])
    package = tmp_path / "evaluator"
    shutil.copytree(Path(module.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    script = """
import runpy
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
import azure.ai.evaluation as sdk
assert Path(sdk.__file__).is_relative_to(Path(sys.argv[1])), sdk.__file__
module = runpy.run_path(sys.argv[2])
cls = module[sys.argv[3]]
assert not module['_SDK_SUPPORTS_DECISION']
config = {'azure_endpoint': 'https://example.openai.azure.com', 'azure_deployment': 'chat'}
original = dict(config)
thresholds = {cls._EVALUATORS[0]['name']: 3.5}
evaluator = cls(config, threshold=thresholds)
assert config == original
thresholds[cls._EVALUATORS[0]['name']] = -1
assert evaluator._threshold_for(cls._EVALUATORS[0]['name']) == 3.5
output = {'llm_output': {member['name']: {'score': member['max'], 'status': 'completed', 'reason': 'test'}
                       for member in cls._EVALUATORS}, 'input_token_count': 30, 'output_token_count': 4,
          'total_token_count': 34, 'model_id': 'actual-model'}
actual_turn = evaluator._flow
actual_conversation = evaluator._multi_turn_flow
async def render_only(timeout, **inputs):
    flow = actual_conversation if 'messages' in inputs else actual_turn
    rendered = flow.render(**inputs)
    assert 'Hello' in str(rendered)
    return output
evaluator._flow = AsyncMock(side_effect=render_only)
evaluator._multi_turn_flow = AsyncMock(side_effect=render_only)
inputs = {'query': 'Hello?', 'response': 'Hello'}
assert evaluator(**inputs)[cls._RESULT_KEY] == 1
assert evaluator(**inputs, tool_definitions=None)[cls._RESULT_KEY] == 1
assert evaluator(messages=[{'role': 'user', 'content': 'Hello?'}, {'role': 'assistant', 'content': 'Hello'}])[
    cls._RESULT_KEY] == 1
for invalid in (3, {'unknown': 1}, {cls._EVALUATORS[0]['name']: True}):
    try:
        cls(config, threshold=invalid)
    except module['EvaluationException']:
        pass
    else:
        raise AssertionError('Invalid threshold accepted')
try:
    cls({'type': 'decision', 'azure_endpoint': 'https://example.services.ai.azure.com',
         'azure_deployment': 'Decision-1'})
except module['EvaluationException'] as exc:
    assert 'parse support' in str(exc)
else:
    raise AssertionError('Decision accepted by released SDK')
print('Released composite package chat/threshold/capability checks passed')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(target)
    result = subprocess.run(
        [sys.executable, "-c", script, str(target), str(package / Path(module.__file__).name),
         evaluator_class.__name__],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
