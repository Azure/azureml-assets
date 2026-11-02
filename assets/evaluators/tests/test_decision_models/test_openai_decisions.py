# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qualify all packaged Decision assets against the shared OpenAI protocol adapter."""

import json
import re
import runpy
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._evaluators._common._decision_protocols import OpenAIDecisionProtocol
from azure.ai.evaluation._evaluators._common._decision_strategy import DecisionRefusalError
from azure.ai.evaluation._legacy.prompty._yaml_utils import load_yaml

from .test_expansion_compatibility import BUILTIN, NAMES, _class, _inputs, _package

ALL_NAMES = (*NAMES, "task_completion", "tool_call_accuracy", "output_quality", "tool_use_quality")
DEPLOYMENT = "decisions-judge"
CONFIG = {"type": "decision", "decision_protocol": "openai", "azure_endpoint": "https://test.openai.azure.com",
          "azure_deployment": DEPLOYMENT, "api_key": "test-only"}


def reply(scenario):
    """Answer the actual encoded rubric, never bypassing SDK translation.

    ``skipped`` refuses every graded question behind a false applicability gate, because a refusal fails a row only
    when the evaluator reads it; ``refusal`` refuses the graded questions of an applicable row.
    """
    async def respond(content, **_):
        request = json.loads(content)
        assert set(request) == {"input", "model", "questions"}
        assert isinstance(json.loads(request["input"]), dict)
        assert request["model"] == DEPLOYMENT
        answers = []
        for question in request["questions"]:
            name = question["name"]
            status = name == "status" or name.endswith("_status")
            if scenario in ("skipped", "refusal") and not status:
                answers.append({"name": name, "type": "refusal"})
            elif question["type"] == "predicate":
                answers.append({"name": name, "type": "predicate", "probability":
                                0 if scenario == "skipped" or (scenario == "failed" and not status) else 0.9})
            else:
                levels = question["levels"]
                # The deflection metric reverses its category scale: index zero is a direct answer.
                index = (len(levels)-1 if scenario == "failed" else 0) if name == "deflection_rate" else (
                    0 if scenario == "failed" else len(levels)-1)
                probabilities = [{"value": i, "label": level["label"], "probability": float(i == index)}
                                 for i, level in enumerate(levels)]
                answers.append({"name": name, "type": "score", "score": index, "confidence": 1,
                                "probabilities": probabilities})
        # Azure OpenAI returns the deployment alias as the model.
        return {"model": DEPLOYMENT, "answers": list(reversed(answers)),
                "usage": {"input_tokens": 10, "output_tokens": 3, "input_tokens_details": {"cached_tokens": 4}}}
    return respond


@pytest.mark.parametrize("name", ALL_NAMES)
@pytest.mark.parametrize("scenario", ["passed", "failed", "skipped", "refusal"])
def test_packaged_openai_decisions(tmp_path, name, scenario):
    """Preserve each asset's verdict, skip, refusal and once-only usage through one OpenAI request."""
    module = runpy.run_path(str(_package(tmp_path, name)))
    evaluator_type = _class(module, name)
    assert evaluator_type._supports_decision_models
    with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", side_effect=AssertionError("Chat loaded")):
        evaluator = evaluator_type(CONFIG)
    post = AsyncMock(side_effect=reply(scenario))
    evaluator._decision_model_runner._post = post
    assert evaluator._decision_model_runner._url.endswith("/openai/v1/decisions")
    if scenario == "refusal":
        with pytest.raises(DecisionRefusalError) as error:
            evaluator(**_inputs(name))
        post.assert_awaited_once()
        assert error.value.properties["total_tokens"] == 13
        provenance = error.value.properties["decision_model"]
        assert provenance["protocol"] == "openai" and provenance["error_kind"] == "model_refusal"
        assert provenance["usage_details"] == {"input_tokens_details": {"cached_tokens": 4}}
        return
    result = evaluator(**_inputs(name))
    post.assert_awaited_once()
    assert result[f"{name}_status"] == ("skipped" if scenario == "skipped" else "completed")
    assert result[f"{name}_passed"] is (None if scenario == "skipped" else scenario == "passed")
    assert result[f"{name}_total_tokens"] == 13
    for member in [member["name"] for member in getattr(evaluator, "_EVALUATORS", ())] or [name]:
        provenance = result[f"{member}_properties"]["decision_model"]
        # Answers are canonical: OpenAI provenance has the Microsoft shape and only names its protocol and model.
        assert provenance["protocol"] == "openai" and provenance["model"] == DEPLOYMENT
        assert provenance["native_status_answer"]["type"] == "noul"
        assert provenance["usage_details"] == {"input_tokens_details": {"cached_tokens": 4}}
        if scenario != "skipped":
            assert provenance["native_answer"]["type"] in ("score", "noul")


def _score_questions(rubric):
    """Yield every ordinal question of a rubric, including each composite member's questions."""
    for question in rubric.values():
        if isinstance(question, dict) and question.get("type") == "score":
            yield question
        elif isinstance(question, dict):
            yield from _score_questions(question)


@pytest.mark.parametrize("rubric", sorted(BUILTIN.glob("*/evaluator/*.decision.yaml")), ids=lambda path: path.name)
def test_rubric_level_references_name_an_encoded_level(rubric):
    """Rubric text such as "imply level 2" must name a level the OpenAI request labels, never a missing one."""
    for question in _score_questions(load_yaml(str(rubric))):
        request = OpenAIDecisionProtocol().encode(DEPLOYMENT, "evidence", {"result": question})
        labels = {level["label"] for level in request["questions"][0]["levels"]}
        for position in re.findall(r"\blevel (\d+)\b", question["instructions"], re.IGNORECASE):
            assert f"Level {position}" in labels


@pytest.mark.parametrize("rubric", ["tool_use_quality.decision.yaml", "tool_use_quality_multi_turn.decision.yaml"])
def test_fabricated_parameters_name_the_fabricated_parameter_level(rubric):
    """The level Tool Call Accuracy names for fabricated parameters is the criterion that describes them."""
    question = load_yaml(str(BUILTIN / "tool_use_quality" / "evaluator" / rubric))["tool_call_accuracy"]["result"]
    assert "Fabricated parameters imply level 2." in question["instructions"]
    request = OpenAIDecisionProtocol().encode(DEPLOYMENT, "evidence", {"tool_call_accuracy": question})
    level = next(level for level in request["questions"][0]["levels"] if level["label"] == "Level 2")
    assert "fabricated parameters" in level["description"]
