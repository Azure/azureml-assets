# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qualify independently packaged expanded assets through SDK public conversion and released wheels."""

import asyncio
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import AsyncMock, patch

import pytest


BUILTIN = Path(__file__).parents[2] / "builtin"
NAMES = (
    "fluency", "coherence", "groundedness", "intent_resolution", "task_adherence",
    "tool_call_success", "tool_input_accuracy", "tool_output_utilization", "tool_selection",
    "customer_satisfaction", "deflection_rate", "relevance", "response_completeness", "retrieval", "similarity",
)
CHAT_CONFIG = {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "chat",
               "api_key": "test-only"}
NATIVE_CONFIG = {**CHAT_CONFIG, "type": "decision", "azure_deployment": "Decision-1"}
USAGE = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def _inputs(name):
    """Turn-level inputs for one asset: a completed tool turn for the tool family, a text pair otherwise."""
    if name.startswith("tool_"):
        return {
            "query": "Look up the weather in Paris.",
            "response": [
                {"role": "assistant", "content": [{"type": "tool_call", "name": "weather",
                 "tool_call_id": "call-1", "arguments": {"city": "Paris"}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": [{"type": "tool_result",
                 "tool_result": {"temperature": 20}}]},
                {"role": "assistant", "content": [{"type": "text", "text": "It is 20 degrees in Paris."}]},
            ],
            "tool_definitions": [{"name": "weather", "description": "Read current temperature.",
                                  "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        }
    result = {"query": "What is the capital of France?", "response": "Paris is the capital of France."}
    if name in ("groundedness", "retrieval"):
        result["context"] = "Paris is the capital of France."
    if name in ("similarity", "response_completeness"):
        result["ground_truth"] = "Paris is the capital of France."
    return result


def _class(module, name):
    """Resolve the evaluator class of a loaded asset module from the asset name."""
    return module["".join(part.title() for part in name.split("_")) + "Evaluator"]


def _package(tmp_path, name):
    """Copy the asset's published evaluator directory into an independent package and return its module path."""
    from azure.ai.evaluation._legacy.prompty._yaml_utils import load_yaml

    spec = load_yaml(str(BUILTIN / name / "spec.yaml"))
    package = tmp_path / name
    shutil.copytree(BUILTIN / name / spec["path"], package, ignore=shutil.ignore_patterns("__pycache__"))
    return package / f"_{name}.py"


def _post_response(name, skipped):
    """Build a transport stub returning a paid native verdict, or a paid skip, for the asset's question types."""
    async def respond(content, *, deadline, execution):
        assert deadline is not None and execution is not None
        body = json.loads(content)
        assert set(body) == {"model", "state", "questions"}
        assert set(body["questions"]) == {"status", name}
        question = body["questions"][name]
        if question["type"] == "noul":
            answer = {"type": "noul", "noul": 0.9}
        else:
            count = len(question["criteria"])
            index = 0 if name == "deflection_rate" else count - 1
            answer = {"type": "score", "score": index, "confidence": 1,
                      "probabilities": {str(i): float(i == index) for i in range(count)}}
        return {"model": "measured-judge", "usage": {"input_tokens": 10, "output_tokens": 3},
                "answers": {"status": {"type": "noul", "noul": 0 if skipped else 1}, name: answer}}
    return respond


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("backend", ["native", "chat"])
@pytest.mark.parametrize("skipped", [False, True])
def test_packaged_expansion_public_usage(tmp_path, name, backend, skipped):
    """A paid verdict or skip retains score semantics, sample usage and once-only run usage after packaging."""
    from azure.ai.evaluation import evaluate

    module = runpy.run_path(str(_package(tmp_path, name)))
    evaluator_type = _class(module, name)
    if backend == "native":
        with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", side_effect=AssertionError("Chat loaded")):
            evaluator = evaluator_type(NATIVE_CONFIG)
        invocation = AsyncMock(side_effect=_post_response(name, skipped))
        evaluator._decision_model_runner._post = invocation
    else:
        evaluator = evaluator_type(CHAT_CONFIG)
        score = 0 if name == "deflection_rate" else 1 if name.startswith("tool_") or name == "task_adherence" else 5
        invocation = AsyncMock(return_value={
            "llm_output": {"score": None if skipped else score, "status": "skipped" if skipped else "completed",
                           "reason": "Fixture judgment", "properties": {"fixture": True}},
            "input_token_count": 10, "output_token_count": 3, "total_token_count": 13, "model_id": "measured-judge",
        })
        evaluator._flow = invocation
    data = tmp_path / "input.jsonl"
    data.write_text(json.dumps(_inputs(name)) + "\n", encoding="utf-8")
    result = evaluate(data=str(data), evaluators={name: evaluator}, _convert_to_aoai_evaluation_result=True)
    invocation.assert_awaited_once()
    row, = result["_evaluation_results_list"]
    metric, = row["results"]
    assert metric["metric"] == name
    assert metric["status"] == ("skipped" if skipped else "completed")
    assert metric["passed"] is (None if skipped else True)
    if skipped:
        assert metric["score"] is None
    elif name == "deflection_rate":
        assert metric["score"] == 0
    elif name.startswith("tool_") or name == "task_adherence":
        assert metric["score"] == 1
    else:
        assert metric["score"] == 5
    assert metric["sample"]["model"] == "measured-judge"
    assert metric["sample"]["usage"] == USAGE
    assert {key: metric["properties"][key] for key in USAGE} == USAGE
    assert result["_evaluation_summary"]["per_model_usage"] == [
        {"model_name": "measured-judge", "invocation_count": 1, **USAGE, "cached_tokens": 0},
    ]


def _exercise_released_asset(path, name):
    """Use the real released base and Prompty loader with only the inference call replaced."""
    from contextlib import ExitStack
    from azure.ai.evaluation._exceptions import EvaluationException

    captured = []
    http_attempts = []

    score = 0 if name == "deflection_rate" else 1 if name.startswith("tool_") or name == "task_adherence" else 5

    async def render_only(flow, timeout, **prepared):
        captured.append(flow.render(**prepared))
        return {"llm_output": {"score": score, "reason": "Fixture", "status": "completed"}}

    def reject_http(*args, **kwargs):
        http_attempts.append(True)
        raise AssertionError("Released SDK qualification attempted unmocked HTTP")

    async def reject_async_http(*args, **kwargs):
        reject_http(*args, **kwargs)

    # The parent pytest transport guard does not propagate to this subprocess.
    with ExitStack() as guards:
        for target in ("azure.core.pipeline.transport.RequestsTransport.send",
                       "requests.adapters.HTTPAdapter.send", "httpx.HTTPTransport.handle_request"):
            guards.enter_context(patch(target, new=reject_http))
        for target in ("azure.core.pipeline.transport.AioHttpTransport.send",
                       "azure.core.pipeline.transport.AsyncioRequestsTransport.send",
                       "httpx.AsyncHTTPTransport.handle_async_request"):
            guards.enter_context(patch(target, new=reject_async_http))
        module = runpy.run_path(path)
        evaluator_type = _class(module, name)
        assert not module["_SDK_SUPPORTS_DECISION"]
        with pytest.raises(EvaluationException, match="Decision"):
            evaluator_type(NATIVE_CONFIG)
        evaluator = evaluator_type(CHAT_CONFIG)
        # Groundedness lazily reloads the query prompt. Intercept the inference
        # class method so the real initial and subsequent prompt loads are covered.
        guards.enter_context(patch.object(type(evaluator._flow), "__call__", new=render_only))
        sync_result = evaluator(**_inputs(name))
        async_result = asyncio.run(evaluator._to_async()(**_inputs(name)))
        assert sync_result == async_result
        assert sync_result[name + "_passed"] is True
        assert sync_result[name + "_status"] == "completed"
        assert len(captured) == 2
        if name == "groundedness":
            assert all(_inputs(name)["query"] in json.dumps(rendered) for rendered in captured)
    assert not http_attempts


@pytest.mark.parametrize("version", ["1.17.0", "1.18.2", "1.18.8", "1.18.9"])
@pytest.mark.parametrize("name", NAMES)
def test_actual_released_wheel_expanded_asset(tmp_path, version, name, released_sdk_target):
    """Every independently copied asset retains sync/async chat with the actual supported released wheels."""
    target = released_sdk_target(version)
    package = _package(tmp_path, name)
    script = """
import runpy
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import azure.ai.evaluation as sdk
from azure.ai.evaluation._version import VERSION
assert VERSION == sys.argv[2], VERSION
assert Path(sdk.__file__).is_relative_to(Path(sys.argv[1])), sdk.__file__
runpy.run_path(sys.argv[3])['_exercise_released_asset'](sys.argv[4], sys.argv[5])
for name, loaded in tuple(sys.modules.items()):
    if name.startswith('azure.ai.evaluation') and getattr(loaded, '__file__', None):
        assert Path(loaded.__file__).is_relative_to(Path(sys.argv[1])), (name, loaded.__file__)
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OTEL_SDK_DISABLED": "true"}
    process = subprocess.run([sys.executable, "-c", script, str(target), version, str(Path(__file__).resolve()),
                              str(package), name], capture_output=True, text=True, timeout=90, env=env)
    assert process.returncode == 0, process.stdout + process.stderr
