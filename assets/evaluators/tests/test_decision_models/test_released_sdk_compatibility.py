# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise released SDK chat compatibility without live model calls."""

import asyncio
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._evaluators import _common
from azure.ai.evaluation._exceptions import EvaluationException


CHAT_CONFIG = {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "test-model"}
DECISION_CONFIG = {"type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
                   "azure_deployment": "Decision-1", "api_key": "test-only"}
BUILTIN = Path(__file__).parents[2] / "builtin"
EVALUATORS = [("task_completion", "TaskCompletionEvaluator"), ("tool_call_accuracy", "ToolCallAccuracyEvaluator")]
STRATEGY_MODULE = "azure.ai.evaluation._evaluators._common._prompty_strategy"
DECISION_MODULE = "azure.ai.evaluation._evaluators._common._decision_strategy"
METADATA = {
    "prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5,
    "finish_reason": "stop", "model": "test", "sample_input": "input", "sample_output": "output",
}
OUTPUT_METADATA = {
    "input_token_count": 2, "output_token_count": 3, "total_token_count": 5,
    "finish_reason": "stop", "model_id": "test", "sample_input": "input", "sample_output": "output",
}
MESSAGES = [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello"}]


@pytest.mark.parametrize("name,class_name", EVALUATORS)
def test_standalone_asset_packages_native_rubrics_without_public_auth(tmp_path, name, class_name):
    """The published evaluator path includes native rubrics and excludes authentication from its public schema."""
    from azure.ai.evaluation._legacy.prompty._yaml_utils import load_yaml
    from jsonschema import ValidationError, validate

    spec = load_yaml(str(BUILTIN / name / "spec.yaml"))
    package = tmp_path / "evaluator"
    shutil.copytree(BUILTIN / name / spec["path"], package, ignore=shutil.ignore_patterns("__pycache__"))
    module = runpy.run_path(str(package / f"_{name}.py"))
    evaluator_class = module[class_name]
    files = [evaluator_class._DECISION_FILE]
    if name == "task_completion":
        files.append(evaluator_class._MULTI_TURN_DECISION_FILE)
    for filename in files:
        questions = module["_load_decision_questions"](str(package / filename), name)
        assert set(questions) == {"status", name}
        assert questions["status"]["type"] == "noul"
        assert questions[name]["type"] == ("noul" if name == "task_completion" else "score")
    schema = spec["initParameterSchema"]
    public_config = {key: value for key, value in DECISION_CONFIG.items() if key != "api_key"}
    validate({"deployment_name": "chat-model"}, schema)
    assert "model_config" not in schema["properties"]
    for selection in ({}, {"model_config": public_config},
                      {"deployment_name": "Decision-1", "model_config": public_config}):
        with pytest.raises(ValidationError):
            validate(selection, schema)
    for auth_field in ("api_key", "credential", "authorization", "headers"):
        with pytest.raises(ValidationError):
            validate({"model_config": {**public_config, auth_field: "not-public"}}, schema)
    with pytest.raises(ValidationError):
        validate({"deployment_name": "chat-model", "model_config": public_config}, schema)
    if name == "task_completion":
        validate({"deployment_name": "Decision-1", "probability_threshold": 0.8}, schema)
        for selection in ({"deployment_name": "Decision-1"},):
            for cutoff in (-0.1, 1.1, "0.8", True):
                with pytest.raises(ValidationError):
                    validate({**selection, "probability_threshold": cutoff}, schema)
        with pytest.raises(ValidationError):
            validate({"deployment_name": "Decision-1", "model_config": public_config,
                      "probability_threshold": 0.8}, schema)


def test_chat_completion_rejects_decision_probability_cutoff_before_loading_prompty():
    """A caller's Decision-only cutoff cannot be ignored on the released chat backend."""
    module = runpy.run_path(str(BUILTIN / "task_completion" / "evaluator" / "_task_completion.py"))
    with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load") as load:
        with pytest.raises(EvaluationException, match="only supported for Decision models"):
            module["TaskCompletionEvaluator"](dict(CHAT_CONFIG), probability_threshold=0.8)
    load.assert_not_called()


def _inputs(name):
    return {"query": "Say hello", "response": "Hello"} if name == "task_completion" else {
        "query": "Weather in Paris?",
        "tool_calls": [{"type": "tool_call", "name": "weather", "tool_call_id": "call-1", "arguments": {}}],
        "tool_definitions": [{"name": "weather", "parameters": {}}],
    }


def _expected(name, score, threshold, *, skipped=False, metadata=None):
    passed = score >= (1 if name == "task_completion" else threshold) if not skipped else None
    result = {
        name: score,
        f"{name}_score": score,
        f"{name}_passed": passed,
        f"{name}_result": "not_applicable" if skipped else "pass" if passed else "fail",
        f"{name}_reason": "Not applicable: reason" if skipped else "reason",
        f"{name}_status": "skipped" if skipped else "completed",
        f"{name}_threshold": threshold,
        f"{name}_properties": None if skipped else {"custom": True, **metadata},
    }
    if name == "task_completion":
        result.update({f"{name}_{key}": value for key, value in metadata.items()})
    return result


def _exercise_chat(module, name, class_name):
    evaluator_class = module[class_name]
    threshold = 0.25 if name == "task_completion" else 3
    original = AsyncMock(side_effect=AssertionError("Original flow must not run"))
    with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", return_value=original):
        evaluator = evaluator_class(dict(CHAT_CONFIG), threshold=threshold)
    if module["PromptyEvaluationStrategy"] is not None:
        assert isinstance(evaluator._strategy, module["PromptyEvaluationStrategy"])
    else:
        assert evaluator._strategy is None
    replacement = AsyncMock()
    evaluator._flow = replacement
    for score in ([0, 0.5, 1, 2] if name == "task_completion" else [1, 2, 3, 5]):
        replacement.return_value = {
            "llm_output": {"score": score, "reason": "reason", "properties": {"custom": True}}, **OUTPUT_METADATA,
        }
        expected = _expected(name, float(score), threshold, metadata=METADATA)
        assert evaluator(**_inputs(name)) == expected
        assert asyncio.run(evaluator._to_async()(**_inputs(name))) == expected
        assert replacement.await_args.kwargs["timeout"] == evaluator._LLM_CALL_TIMEOUT

    replacement.return_value = {"llm_output": {"status": "skipped", "reason": "reason"}, **OUTPUT_METADATA}
    empty_metadata = {key: 0 if key.endswith("tokens") else "" for key in METADATA}
    expected = _expected(name, None, threshold, skipped=True, metadata=empty_metadata)
    assert evaluator(**_inputs(name)) == expected
    assert asyncio.run(evaluator._to_async()(**_inputs(name))) == expected

    for output in (["invalid"], {"llm_output": "{\"score\": 1}"}, "invalid", None):
        replacement.return_value = output
        with pytest.raises(EvaluationException, match="invalid output"):
            evaluator(**_inputs(name))
    if name == "tool_call_accuracy":
        for score in (None, 0, -1, 6, "invalid", float("nan")):
            replacement.return_value = {"llm_output": {"score": score}}
            with pytest.raises(EvaluationException, match="Invalid score"):
                evaluator(**_inputs(name))
        replacement.return_value = {"llm_output": {}}
        with pytest.raises(EvaluationException, match="Invalid score"):
            evaluator(**_inputs(name))
    else:
        replacement.return_value = {"llm_output": {}}
        assert evaluator(**_inputs(name))["task_completion_score"] == 0
        replacement.return_value = {"llm_output": {"score": 1}}
        replacement.reset_mock()
        multi = AsyncMock(return_value={
            "llm_output": {"score": 0, "reason": "reason", "properties": {"custom": True}}, **OUTPUT_METADATA,
        })
        evaluator._multi_turn_flow = multi
        assert evaluator(messages=MESSAGES) == _expected(name, 0.0, threshold, metadata=METADATA)
        assert asyncio.run(evaluator._to_async()(messages=MESSAGES))["task_completion_result"] == "fail"
        replacement.assert_not_awaited()
        assert "User turn 1:" in multi.await_args.kwargs["messages"]
        assert "Agent turn 1:" in multi.await_args.kwargs["messages"]
        multi.return_value = {"llm_output": {"status": "skipped", "reason": "reason"}}
        assert evaluator(messages=MESSAGES) == _expected(name, None, threshold, skipped=True, metadata=empty_metadata)
        for level in ("conversation", "turn"):
            with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", return_value=original):
                routed = evaluator_class(dict(CHAT_CONFIG), evaluation_level=level)
            routed._flow = AsyncMock(return_value={"llm_output": {"score": 1}})
            routed._multi_turn_flow = AsyncMock(return_value={"llm_output": {"score": 0}})
            inputs = _inputs(name) if level == "conversation" else {"messages": MESSAGES}
            assert routed(**inputs)["task_completion_score"] == (0 if level == "conversation" else 1)
            selected = routed._multi_turn_flow if level == "conversation" else routed._flow
            unused = routed._flow if level == "conversation" else routed._multi_turn_flow
            selected.assert_awaited_once()
            unused.assert_not_awaited()
    original.assert_not_awaited()
    if not module["_SDK_SUPPORTS_DECISION"]:
        with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load") as load:
            with pytest.raises(EvaluationException, match="version with decision model support"):
                evaluator_class(dict(DECISION_CONFIG))
            load.assert_not_called()


@pytest.mark.parametrize("name,class_name", EVALUATORS)
@pytest.mark.parametrize("capabilities", ["shared", "old_base", "old_sdk", "no_strategy", "no_decision",
                                          "no_strategy_symbol", "no_decision_symbol"])
def test_released_sdk_prompt_strategy_and_decision_rejection(monkeypatch, name, class_name, capabilities):
    """Preserve chat evaluation and reject unsupported Decision configurations across SDK capabilities."""
    if capabilities in ("old_base", "old_sdk"):
        monkeypatch.delattr(_common, "ModelEvaluatorBase", raising=False)
    if capabilities in ("old_sdk", "no_strategy"):
        monkeypatch.setitem(sys.modules, STRATEGY_MODULE, None)
    if capabilities == "no_decision":
        monkeypatch.setitem(sys.modules, DECISION_MODULE, None)
    if capabilities == "no_strategy_symbol":
        if sys.modules.get(STRATEGY_MODULE) is None:
            pytest.skip("Installed SDK has no shared Prompty strategy to remove")
        monkeypatch.delattr(sys.modules[STRATEGY_MODULE], "PromptyEvaluationStrategy")
    if capabilities == "no_decision_symbol":
        if sys.modules.get(DECISION_MODULE) is None:
            pytest.skip("Installed SDK has no Decision strategy to remove")
        monkeypatch.delattr(sys.modules[DECISION_MODULE], "DecisionEvaluationStrategy")
    module = runpy.run_path(str(BUILTIN / name / "evaluator" / f"_{name}.py"))
    if capabilities in ("old_base", "old_sdk", "no_decision", "no_decision_symbol"):
        assert module["_SDK_SUPPORTS_DECISION"] is False
    base = getattr(_common, "ModelEvaluatorBase", _common.PromptyEvaluatorBase)
    assert issubclass(module[class_name], base)
    _exercise_chat(module, name, class_name)


@pytest.mark.parametrize("name,class_name", EVALUATORS)
@pytest.mark.parametrize("error", [ImportError("Broken SDK module"), ModuleNotFoundError(name="broken_dependency")])
def test_broken_strategy_import_is_not_hidden(monkeypatch, name, class_name, error):
    """Propagate broken strategy imports rather than treating them as unavailable SDK features."""
    import builtins

    original_import = builtins.__import__

    def import_with_broken_strategy(module_name, *args, **kwargs):
        if module_name == STRATEGY_MODULE:
            raise error
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_broken_strategy)
    with pytest.raises(type(error)) as caught:
        runpy.run_path(str(BUILTIN / name / "evaluator" / f"_{name}.py"))
    assert caught.value is error


@pytest.mark.parametrize("version", ["1.17.0", "1.18.2"])
@pytest.mark.parametrize("name,class_name", EVALUATORS)
def test_actual_released_wheel_standalone_asset(tmp_path, version, name, class_name, released_sdk_target):
    """Verify standalone asset chat evaluation with an isolated released SDK installation."""
    target = released_sdk_target(version)
    package = tmp_path / "evaluator"
    shutil.copytree(BUILTIN / name / "evaluator", package, ignore=shutil.ignore_patterns("__pycache__"))
    script = """
import runpy
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import azure.ai.evaluation as sdk
from azure.ai.evaluation._version import VERSION
assert Path(sdk.__file__).is_relative_to(Path(sys.argv[1])), sdk.__file__
assert VERSION == sys.argv[6], VERSION
tests = runpy.run_path(sys.argv[2])
module = runpy.run_path(sys.argv[3])
assert module['PromptyEvaluationStrategy'] is None
assert not module['_SDK_SUPPORTS_DECISION']
tests['_exercise_chat'](module, sys.argv[4], sys.argv[5])
print('Released SDK standalone sync/async chat checks passed')
"""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script, str(target), str(Path(__file__).resolve()),
         str(package / f"_{name}.py"), name, class_name, version],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("version", ["1.17.0", "1.18.2"])
@pytest.mark.parametrize("configured", [False, True])
def test_released_sdk_absence_is_an_actionable_skip(tmp_path, monkeypatch, released_sdk_target, version, configured):
    """Explain missing opt-in or isolated SDK installations without failing clean machines."""
    if configured:
        monkeypatch.setenv("AZUREML_COMPAT_SDK_ROOT", str(tmp_path))
    else:
        monkeypatch.delenv("AZUREML_COMPAT_SDK_ROOT", raising=False)
    with pytest.raises(pytest.skip.Exception, match="AZUREML_COMPAT_SDK_ROOT"):
        released_sdk_target(version)
