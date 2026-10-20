# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline transport-level coverage for native standalone quality evaluators."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.fluency.evaluator._fluency import FluencyEvaluator
from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator
from ...builtin.intent_resolution.evaluator._intent_resolution import IntentResolutionEvaluator
from ...builtin.task_adherence.evaluator._task_adherence import TaskAdherenceEvaluator
from ..test_evaluators_behavior import common_tool_test_data as data


EVALUATORS = [FluencyEvaluator, CoherenceEvaluator, GroundednessEvaluator,
              IntentResolutionEvaluator, TaskAdherenceEvaluator]
CONVERSATION_EVALUATORS = [CoherenceEvaluator, GroundednessEvaluator, TaskAdherenceEvaluator]
LIKERT_EVALUATORS = EVALUATORS[:-1]
CONFIG = {"type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
          "azure_deployment": "Decision-1", "api_key": "test-only"}
CHAT = {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "chat",
        "api_key": "test-only"}
PROMPTY_LOAD = "azure.ai.evaluation._legacy.prompty.AsyncPrompty.load"
MESSAGES = [{"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Latest question"},
            {"role": "assistant", "content": "Latest answer"}]
USAGE = {"input_token_count": 11, "output_token_count": 7, "total_token_count": 18,
         "model_id": "actual-chat", "finish_reason": "stop", "sample_input": "input", "sample_output": "output"}


def native(name, *, level=5, probability=1, status=1, usage=None):
    """Create a native transport response; native Likert levels are zero-based."""
    answer = ({"type": "noul", "noul": probability} if name == "task_adherence" else {
        "type": "score", "score": level - 1, "confidence": 1,
        "probabilities": {str(i): int(i == level - 1) for i in range(5)},
    })
    return {"model": "actual-native", "answers": {
        "status": {"type": "noul", "noul": status}, name: answer,
    }, "usage": usage if usage is not None else {"input_tokens": 11, "output_tokens": 7}}


def inputs_for(cls):
    """Return valid standalone inputs without relying on a composite contract."""
    if cls is FluencyEvaluator:
        return {"response": "The museum opens at noon."}
    if cls is GroundednessEvaluator:
        return {"response": "The museum opens at noon.", "context": "Museum opening time: noon."}
    return {"query": "When does the museum open?", "response": "The museum opens at noon."}


def stub(evaluator, response=None, **kwargs):
    """Replace only the SDK transport, retaining preparation, scoring and validation."""
    evaluator._model_runner._post = AsyncMock(return_value=response, **kwargs)
    return evaluator._model_runner._post


def evidence(post, index=-1):
    content, = post.call_args_list[index].args
    request = json.loads(content)
    assert set(request) == {"model", "state", "questions"}
    assert request["model"] == CONFIG["azure_deployment"]
    return request["state"], request["questions"]


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_native_happy_path_without_prompty(cls, asynchronous):
    """Native construction never loads a chat prompt and retains scores and measured provenance."""
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Prompty must not load")):
        evaluator = cls(CONFIG)
        name = evaluator._result_key
        post = stub(evaluator, native(name))
        result = (asyncio.run(evaluator._to_async()(**inputs_for(cls))) if asynchronous
                  else evaluator(**inputs_for(cls)))
    score = 1 if cls is TaskAdherenceEvaluator else 5
    assert result[name] == result[f"{name}_score"] == score
    assert result[f"{name}_threshold"] == (0 if cls is TaskAdherenceEvaluator else 3)
    assert result[f"{name}_result"] == "pass"
    assert result[f"{name}_passed"] is True
    assert result[f"{name}_total_tokens"] == 18
    assert result[f"{name}_model"] == "actual-native"
    properties = result[f"{name}_properties"]
    assert properties["total_tokens"] == 18
    assert properties["decision_model"]["deployment"] == "Decision-1"
    assert "No model rationale" in result[f"{name}_reason"]
    assert not {"finish_reason", "sample_input", "sample_output"} & properties.keys()
    _, questions = evidence(post)
    assert set(questions) == {"status", name}
    assert questions[name]["type"] == ("noul" if cls is TaskAdherenceEvaluator else "score")
    assert "data" in questions[name]["instructions"]


@pytest.mark.parametrize("cls", EVALUATORS)
def test_model_skip_keeps_usage_without_reading_unusable_score(cls):
    """A paid applicability skip keeps provenance and ignores its unscorable answer."""
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    response = native(name, status=0.49999)
    response["answers"][name] = ({"type": "noul", "noul": "unusable"} if cls is TaskAdherenceEvaluator
                                 else {"type": "score", "probabilities": "unusable"})
    post = stub(evaluator, response)
    result = evaluator(**inputs_for(cls))
    post.assert_awaited_once()
    assert result[name] is result[f"{name}_score"] is result[f"{name}_passed"] is None
    assert result[f"{name}_status"] == "skipped"
    assert result[f"{name}_result"] == "not_applicable"
    assert result[f"{name}_total_tokens"] == 18
    assert result[f"{name}_properties"]["model"] == "actual-native"


@pytest.mark.parametrize("cls", LIKERT_EVALUATORS)
@pytest.mark.parametrize("level,passed", [(1, False), (3, False), (4, True), (5, True)])
def test_likert_threshold_boundaries_and_status_cutoff(cls, level, passed):
    """Map the full score range using an inclusive public threshold and status cutoff."""
    evaluator = cls(CONFIG, threshold=4)
    name = evaluator._result_key
    stub(evaluator, native(name, level=level, status=0.5))
    result = evaluator(**inputs_for(cls))
    assert result[name] == level
    assert result[f"{name}_passed"] is passed
    assert result[f"{name}_threshold"] == 4


@pytest.mark.parametrize("cls", LIKERT_EVALUATORS)
def test_likert_tie_selects_lower_level(cls):
    """Equal top probabilities select the lower rating, not the provider's rounded mean."""
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    response = native(name)
    response["answers"][name].update(score=2.5, confidence=0.5,
                                     probabilities={"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0})
    stub(evaluator, response)
    assert evaluator(**inputs_for(cls))[name] == 3


@pytest.mark.parametrize("top_level", [False, True])
@pytest.mark.parametrize("probability,score", [(0.79999, 0), (0.8, 1)])
def test_adherence_probability_cutoff_and_legacy_public_threshold(probability, score, top_level):
    """The probability cutoff affects the binary score; public threshold zero never makes a failure pass."""
    config = {**CONFIG, "probability_threshold": 0.75 if top_level else 0.8}
    original = deepcopy(config)
    evaluator = TaskAdherenceEvaluator(config, **({"probability_threshold": 0.8} if top_level else {}))
    stub(evaluator, native("task_adherence", probability=probability))
    result = evaluator(**inputs_for(TaskAdherenceEvaluator))
    assert result["task_adherence"] == score
    assert result["task_adherence_passed"] is bool(score)
    assert result["task_adherence_threshold"] == 0
    assert config == original


@pytest.mark.parametrize("cutoff", [0, 1])
def test_adherence_probability_schema_accepts_inclusive_endpoints(cutoff):
    """The packaged schema and constructor accept the same probability endpoints."""
    from azure.ai.evaluation._legacy.prompty._yaml_utils import load_yaml
    from jsonschema import validate

    spec = load_yaml(str(Path(__file__).parents[2] / "builtin" / "task_adherence" / "spec.yaml"))
    validate({"deployment_name": "Decision-1", "probability_threshold": cutoff}, spec["initParameterSchema"])
    evaluator = TaskAdherenceEvaluator(CONFIG, probability_threshold=cutoff)
    stub(evaluator, native("task_adherence", probability=cutoff))
    assert evaluator(**inputs_for(TaskAdherenceEvaluator))["task_adherence"] == 1


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, float("nan"), float("inf"), "0.8", True])
def test_adherence_invalid_probability_cutoff_rejected(cutoff):
    """The explicit override uses SDK validation rather than silently falling back to the default."""
    with pytest.raises(EvaluationException):
        TaskAdherenceEvaluator(CONFIG, probability_threshold=cutoff)


def test_adherence_probability_cutoff_requires_native_selection():
    """A Decision-only control cannot silently change or initialize a chat evaluator."""
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Chat must not initialize")):
        with pytest.raises(EvaluationException, match="only supported for Decision"):
            TaskAdherenceEvaluator(CHAT, probability_threshold=0.8)


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("bad", ["empty", "status", "score"])
def test_invalid_native_answers_raise_with_known_usage(cls, bad):
    """Malformed native output is an error, retaining usage if a response reports it."""
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    response = native(name)
    if bad == "empty":
        response = {}
    elif bad == "status":
        response["answers"]["status"]["noul"] = 1.1
    else:
        response["answers"][name] = ({"type": "noul", "noul": -1} if cls is TaskAdherenceEvaluator
                                     else {"type": "score", "score": 8, "confidence": 1,
                                           "probabilities": {"0": 2, "1": 0, "2": 0, "3": 0, "4": 0}})
    stub(evaluator, response)
    with pytest.raises(EvaluationException) as caught:
        evaluator(**inputs_for(cls))
    if bad != "empty":
        assert caught.value.properties["total_tokens"] == 18


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("invalid", ["missing_endpoint", "legacy_backend"])
def test_invalid_native_configuration_never_falls_back(cls, invalid):
    """A rejected native or legacy selection cannot initialize a chat flow."""
    config = ({key: value for key, value in CONFIG.items() if key != "azure_endpoint"}
              if invalid == "missing_endpoint" else {"backend": "decision", "endpoint": "https://example.com"})
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Unexpected chat fallback")):
        with pytest.raises(EvaluationException):
            cls(config)


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("shape", ["text", "messages", "tool_messages"])
def test_native_and_chat_receive_identical_prepared_inputs(cls, shape):
    """The native transport consumes exactly the legacy flow's prepared evidence."""
    inputs = inputs_for(cls)
    if shape == "messages":
        inputs = {"messages": deepcopy(MESSAGES)}
        if cls is GroundednessEvaluator:
            inputs["context"] = "The first answer and latest answer are supplied evidence."
    elif shape == "tool_messages":
        inputs = {"query": deepcopy(data.FILE_SEARCH_QUERY), "response": deepcopy(data.FILE_SEARCH_RESPONSE),
                  "tool_definitions": deepcopy(data.FILE_SEARCH_TOOL_DEFINITIONS)}
    original = deepcopy(inputs)
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    post = stub(evaluator, native(name))
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Native lazy Prompty load")):
        evaluator(**inputs)
    chat = cls(CHAT)
    response = {"llm_output": {"score": 1 if cls is TaskAdherenceEvaluator else 5,
                               "reason": "Fixture judgment", "status": "completed"}}
    flow = AsyncMock(return_value=response)
    chat._flow = flow
    if hasattr(chat, "_multi_turn_flow"):
        chat._multi_turn_flow = flow
    # Groundedness lazily selects its query prompt for extracted tool context.
    with patch(PROMPTY_LOAD, return_value=flow):
        chat(**deepcopy(inputs))
    flow.assert_awaited_once()
    prepared = {key: value for key, value in flow.call_args.kwargs.items() if key != "timeout"}
    state, _ = evidence(post)
    assert state == json.loads(json.dumps(prepared))
    assert inputs == original


@pytest.mark.parametrize("cls", CONVERSATION_EVALUATORS)
@pytest.mark.parametrize("level", ["turn", "conversation"])
def test_existing_explicit_evaluation_levels_route_same_evidence(cls, level):
    """Existing explicit levels select latest-turn evidence or the entire transcript."""
    evaluator = cls(CONFIG, evaluation_level=level)
    name = evaluator._result_key
    post = stub(evaluator, native(name))
    inputs = {"messages": deepcopy(MESSAGES)}
    if cls is GroundednessEvaluator:
        inputs["context"] = "Evidence for the first and latest answer."
    evaluator(**inputs)
    state, questions = evidence(post)
    if level == "conversation":
        assert "First answer" in state["messages"] and "Latest answer" in state["messages"]
    else:
        assert "Latest answer" in json.dumps(state["response"])
        assert "First answer" not in json.dumps(state["response"])
    assert len(questions) == 2


@pytest.mark.parametrize("cls", EVALUATORS)
def test_intermediate_response_skips_before_transport(cls):
    """Intermediate tool-only responses are not paid evaluations."""
    evaluator = cls(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Intermediate response reached transport"))
    inputs = {"query": deepcopy(data.LOCAL_CALLS_QUERY), "response": deepcopy(data.LOCAL_CALLS_RESPONSE[:1]),
              "tool_definitions": deepcopy(data.LOCAL_CALLS_TOOL_DEFINITIONS)}
    if cls is GroundednessEvaluator:
        inputs["context"] = "The known supporting source."
    result = evaluator(**inputs)
    name = evaluator._result_key
    assert result[f"{name}_status"] == "skipped"
    assert result[f"{name}_total_tokens"] == 0
    assert result[f"{name}_properties"] is None
    post.assert_not_called()


@pytest.mark.parametrize("cls", EVALUATORS)
@pytest.mark.parametrize("conversation", [False, True])
def test_paid_chat_skip_preserves_measured_usage(cls, conversation):
    """Skipped model judgments retain charged tokens and supplied properties in both existing paths."""
    chat = cls(CHAT)
    name = chat._result_key
    output = {"llm_output": {"score": None, "status": "skipped", "reason": "Not enough evidence",
                              "properties": {"custom": True}}, **USAGE}
    flow = AsyncMock(return_value=output)
    chat._flow = flow
    inputs = inputs_for(cls)
    if conversation:
        inputs = {"messages": deepcopy(MESSAGES)}
        if cls is GroundednessEvaluator:
            inputs["context"] = "Grounding evidence."
        if cls in CONVERSATION_EVALUATORS:
            chat._multi_turn_flow = flow
    result = chat(**inputs)
    assert result[name] is None
    assert result[f"{name}_result"] == "not_applicable"
    assert result[f"{name}_total_tokens"] == 18
    assert result[f"{name}_model"] == "actual-chat"
    properties = result[f"{name}_properties"]
    assert properties["custom"] is True
    assert properties["total_tokens"] == 18
    assert properties["sample_input"] == "input"
    assert output["llm_output"]["properties"] == {"custom": True}


@pytest.mark.parametrize("cls", EVALUATORS)
def test_split_conversation_usage_is_summed(cls):
    """The native SDK aggregation sums paid invocations while continuing to average scores."""
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    responses = [native(name, level=3), native(name, level=5, usage={"input_tokens": 19, "output_tokens": 3})]
    post = stub(evaluator, side_effect=responses)
    messages = deepcopy(MESSAGES)
    if cls is GroundednessEvaluator:
        for message in messages:
            if message["role"] == "assistant":
                message["context"] = "This response has known support."
    result = evaluator(conversation={"messages": messages})
    assert post.await_count == 2
    assert result[name] == (1 if cls is TaskAdherenceEvaluator else 4)
    assert result[f"{name}_prompt_tokens"] == 30
    assert result[f"{name}_completion_tokens"] == 10
    assert result[f"{name}_total_tokens"] == 40
    assert result[f"{name}_model"] == "actual-native"


@pytest.mark.parametrize("cls", EVALUATORS)
def test_later_split_turn_failure_retains_prior_paid_usage(cls):
    """A failed later turn reaches the fixed SDK failure hook instead of losing successful-turn charges."""
    evaluator = cls(CONFIG)
    name = evaluator._result_key
    bad = native(name, usage={"input_tokens": 19, "output_tokens": 3})
    bad["answers"]["status"]["noul"] = 2
    post = stub(evaluator, side_effect=[native(name), bad])
    messages = deepcopy(MESSAGES)
    if cls is GroundednessEvaluator:
        for message in messages:
            if message["role"] == "assistant":
                message["context"] = "This response has known support."
    with pytest.raises(EvaluationException) as caught:
        evaluator(conversation={"messages": messages})
    assert post.await_count == 2
    properties = caught.value.properties
    assert properties["prompt_tokens"] == 30
    assert properties["completion_tokens"] == 10
    assert properties["total_tokens"] == 40
    assert properties["decision_model"]["result_usage_scope"] == "reported_turns_only"
    assert "execution" not in properties["decision_model"]


def test_groundedness_preserves_context_with_and_without_query():
    """Separate standalone rubrics preserve their different level-three meanings and source context."""
    evaluator = GroundednessEvaluator(CONFIG)
    post = stub(evaluator, native("groundedness", level=3))
    evaluator(response="The company is growing.", context="Profits increased 20%.")
    state, questions = evidence(post)
    assert state["context"] == "Profits increased 20%."
    assert "Accurate but vague" in questions["groundedness"]["criteria"][2]
    evaluator(query="How much did profits rise?", response="Could you clarify?", context="Profits increased 20%.")
    state, questions = evidence(post)
    assert state["context"] == "Profits increased 20%."
    assert "Nothing to ground" in questions["groundedness"]["criteria"][2]


def test_groundedness_conversation_chat_renders_explicit_context():
    """Supplied conversation context reaches both judges and is visible in the rendered chat prompt."""
    inputs = {"messages": deepcopy(MESSAGES), "context": "Dedicated source: museum opens at noon."}
    evaluator = GroundednessEvaluator(CONFIG)
    post = stub(evaluator, native("groundedness"))
    evaluator(**inputs)
    state, _ = evidence(post)
    chat = GroundednessEvaluator(CHAT)
    flow = chat._multi_turn_flow
    rendered = []

    async def render_only(timeout, **prepared):
        rendered.append(json.dumps(flow.render(**prepared)))
        assert prepared == state
        return {"llm_output": {"score": 5, "status": "completed", "reason": "Fixture"}}

    chat._multi_turn_flow = AsyncMock(side_effect=render_only)
    chat(**inputs)
    assert inputs["context"] in rendered[0]


def test_fluency_json_message_response_only_scores_assistant_text():
    """JSON-encoded tool transcripts keep the existing final-text extraction before native transport."""
    evaluator = FluencyEvaluator(CONFIG)
    post = stub(evaluator, native("fluency"))
    evaluator(response=json.dumps(data.LOCAL_CALLS_RESPONSE))
    state, _ = evidence(post)
    assert state["response"] == data.LOCAL_CALLS_IR_EXPECTED_FLOW_RESPONSE
    assert "function_call" not in state["response"]


@pytest.mark.parametrize("cls", EVALUATORS)
def test_standalone_packages_include_native_rubrics(tmp_path, cls):
    """An independently copied package can construct native execution without sibling runtime files."""
    name = cls._RESULT_KEY
    source = Path(__file__).parents[2] / "builtin" / name / "evaluator"
    package = tmp_path / "evaluator"
    shutil.copytree(source, package, ignore=shutil.ignore_patterns("__pycache__"))
    module = runpy.run_path(str(package / f"_{name}.py"))
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Prompty must not load")):
        evaluator = module[cls.__name__](CONFIG)
    assert evaluator._decision.questions[name]["type"] in {"score", "noul"}
    if cls in CONVERSATION_EVALUATORS:
        assert evaluator._multi_turn_decision.questions["status"]["type"] == "noul"


@pytest.mark.parametrize("cls", EVALUATORS)
def test_unsupported_sdk_rejects_native_clearly_without_chat(cls):
    """A package's compatibility guard rejects native setup when the old SDK lacks the required strategy."""
    module_name = cls.__module__
    with patch(f"{module_name}._SDK_SUPPORTS_DECISION", False), patch(
        PROMPTY_LOAD, side_effect=AssertionError("Unexpected chat fallback")
    ):
        with pytest.raises(EvaluationException, match="version with decision model support"):
            cls(CONFIG)
