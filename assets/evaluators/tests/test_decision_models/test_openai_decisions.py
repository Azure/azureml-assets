# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qualify all packaged Decision assets against the shared OpenAI protocol adapter."""

import json
import runpy
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from .test_expansion_compatibility import NAMES, _class, _inputs, _package

ALL_NAMES = (*NAMES, "task_completion", "tool_call_accuracy", "output_quality", "tool_use_quality")
CONFIG = {"type": "decision", "decision_protocol": "openai", "azure_endpoint": "https://test.openai.azure.com",
          "azure_deployment": "decisions-judge", "api_key": "test-only"}


def reply(scenario):
    """Return contract fixtures for the actual encoded rubric, never bypassing SDK translation."""
    async def respond(content, **_):
        request = json.loads(content)
        assert set(request) == {"input", "model", "questions"}
        assert isinstance(json.loads(request["input"]), dict)
        assert request["model"] == CONFIG["azure_deployment"]
        answers = []
        for question in request["questions"]:
            name = question["name"]
            status = name == "status" or name.endswith("_status")
            if scenario == "refusal" and not status:
                answers.append({"name": name, "type": "refusal"})
            elif question["type"] == "predicate":
                answers.append({"name": name, "type": "predicate", "probability":
                                0 if (scenario == "skipped" and status) or (scenario == "failed" and not status) else 0.9})
            else:
                levels = question["levels"]
                # The deflection metric reverses its category scale: index zero is a direct answer.
                index = (len(levels)-1 if scenario == "failed" else 0) if name == "deflection_rate" else (
                    0 if scenario == "failed" else len(levels)-1)
                answers.append({"name": name, "type": "score", "score": index, "confidence": 1,
                    "probabilities": [{"value": i, "label": level["label"], "probability": float(i == index)}
                                      for i, level in enumerate(levels)]})
        return {"model": "gpt-6-luna-2026-09-22", "answers": list(reversed(answers)),
                "usage": {"input_tokens": 10, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 4}}}
    return respond


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("scenario", ["passed", "failed", "skipped", "refusal"])
def test_packaged_openai_decisions(tmp_path, name, scenario):
    """Preserve each asset's score/verdict, explicit refusals and request usage after packaging."""
    module = runpy.run_path(str(_package(tmp_path, name)))
    evaluator_type = _class(module, name)
    assert "openai" in evaluator_type._supported_decision_protocols
    with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", side_effect=AssertionError("Chat loaded")):
        evaluator = evaluator_type(CONFIG)
    evaluator._decision_model_runner._post = AsyncMock(side_effect=reply(scenario))
    assert evaluator._decision_model_runner._url.endswith("/openai/v1/decisions")
    if scenario == "refusal":
        with pytest.raises(EvaluationException, match="refused") as error:
            evaluator(**_inputs(name))
        assert error.value.properties["total_tokens"] == 13
        return
    result = evaluator(**_inputs(name))
    assert result[f"{name}_status"] == ("skipped" if scenario == "skipped" else "completed")
    assert result[f"{name}_passed"] is (None if scenario == "skipped" else scenario == "passed")
    assert result[f"{name}_total_tokens"] == 13
    if name not in ("output_quality", "tool_use_quality"):
        provenance = result[f"{name}_properties"]["decision_model"]
        assert provenance["protocol"] == "openai"
        assert provenance["native_status_answer"]["type"] == "predicate"
        if scenario != "skipped":
            assert provenance["native_answer"]["type"] in ("score", "predicate")
