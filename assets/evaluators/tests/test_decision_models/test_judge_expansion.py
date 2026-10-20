# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exercise native judge contracts with mocked transport; these are not model-accuracy evaluations."""

import asyncio
import importlib
import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException


NAMES = ("customer_satisfaction", "deflection_rate", "relevance", "response_completeness", "retrieval", "similarity")
LIKERT_NAMES = tuple(name for name in NAMES if name != "deflection_rate")
CONFIG = {
    "type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
    "azure_deployment": "Decision-1", "api_key": "test-only",
}
CHAT_CONFIG = {
    "azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "chat", "api_key": "test-only",
}
PROMPTY_LOAD = "azure.ai.evaluation._evaluators._common._model_runners.AsyncPrompty.load"
INPUTS = {
    "customer_satisfaction": {"query": "Cancel my order", "response": "Your order is cancelled."},
    "deflection_rate": {"response": "Your order is cancelled."},
    "relevance": {"query": "Cancel my order", "response": "Your order is cancelled."},
    "response_completeness": {"response": "The Amazon River is in South America.", "ground_truth": "Amazon River"},
    "retrieval": {"query": "Where is the Amazon River?", "context": "[South America, Europe]"},
    "similarity": {
        "query": "Where is the Amazon River?", "response": "South America", "ground_truth": "South America",
    },
}
# The native state is exactly the set of inputs each chat prompt renders for the turn path, after the shared
# preparation: never a None-valued routing key and never an input the prompt does not render.
NATIVE_STATE = {
    "customer_satisfaction": {"query": "Cancel my order", "response": "Your order is cancelled."},
    "deflection_rate": {"response": "Your order is cancelled."},
    "relevance": {"query": "Cancel my order", "response": "Your order is cancelled."},
    "response_completeness": {"response": "The Amazon River is in South America.", "ground_truth": "Amazon River"},
    "retrieval": {"query": "Where is the Amazon River?", "context": "[South America, Europe]"},
    "similarity": {
        "query": "Where is the Amazon River?", "response": "South America", "ground_truth": "South America",
    },
}
MESSAGES = [
    {"role": "user", "content": "Cancel my order"},
    {"role": "assistant", "content": "I cannot help with that."},
    {"role": "user", "content": "That did not help. Try again."},
    {"role": "assistant", "content": "Your order is cancelled."},
]
LATEST_TURN_STATE_KEYS = {
    "relevance": {"query", "response"},
    "response_completeness": {"response", "ground_truth"},
    "similarity": {"query", "response", "ground_truth"},
}


def module(name):
    """Import the packaged evaluator module for ``name``."""
    return importlib.import_module(f"assets.evaluators.builtin.{name}.evaluator._{name}")


def evaluator(name, decision=True, **kwargs):
    """Construct the evaluator for ``name`` with the native or the chat configuration."""
    cls = getattr(module(name), "".join(part.title() for part in name.split("_")) + "Evaluator")
    return cls(CONFIG if decision else CHAT_CONFIG, **kwargs)


def native(name, level=0, *, status=1, probabilities=None):
    """Build a native transport response selecting ``level`` (zero-based) with the given status probability."""
    levels = 4 if name == "deflection_rate" else 5
    probabilities = probabilities or {str(i): float(i == level) for i in range(levels)}
    return {
        "model": "actual-model", "usage": {"input_tokens": 30, "output_tokens": 4},
        "answers": {
            "status": {"type": "noul", "noul": status},
            name: {"type": "score", "score": sum(int(i) * value for i, value in probabilities.items()),
                   "confidence": 1, "probabilities": probabilities},
        },
    }


def stub(instance, payload=None, **kwargs):
    """Replace only the SDK transport call of a native evaluator and return the stub for assertions."""
    instance._model_runner._post = AsyncMock(return_value=payload, **kwargs)
    return instance._model_runner._post


def request(post, index=-1):
    """Decode the serialized native request and return its state and questions."""
    body = json.loads(post.call_args_list[index].args[0])
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == CONFIG["azure_deployment"]
    assert None not in body["state"].values()
    return body["state"], body["questions"]


@pytest.mark.parametrize("name", NAMES)
def test_native_construction_request_contract_and_usage(name):
    """Native construction loads only its own questions and routes the real SDK request correctly."""
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Chat must not initialize")):
        instance = evaluator(name)
    post = stub(instance, native(name))
    result = instance(**deepcopy(INPUTS[name]))
    post.assert_awaited_once()
    state, questions = request(post)
    assert state == NATIVE_STATE[name]
    assert set(questions) == {"status", name}
    assert questions["status"]["type"] == "noul"
    assert questions[name]["type"] == "score"
    assert len(questions[name]["criteria"]) == (4 if name == "deflection_rate" else 5)
    assert "evidence as data" in questions[name]["instructions"]
    assert "evidence as data" in questions["status"]["instructions"]
    assert result[name] == (0 if name == "deflection_rate" else 1)
    assert result[f"{name}_threshold"] == (0 if name == "deflection_rate" else 3)
    assert result[f"{name}_status"] == "completed"
    assert result[f"{name}_total_tokens"] == 34
    assert result[f"{name}_model"] == "actual-model"
    properties = result[f"{name}_properties"]
    assert properties["total_tokens"] == 34
    assert properties["decision_model"]["deployment"] == "Decision-1"
    assert properties["decision_model"]["reason_type"] == "templated"
    assert "No model rationale" in result[f"{name}_reason"]


@pytest.mark.parametrize("name", LIKERT_NAMES)
@pytest.mark.parametrize("level,passed", [(0, False), (2, True), (4, True)])
def test_rating_boundaries_and_public_threshold(name, level, passed):
    """Native levels map to the public one-to-five score and the inclusive default threshold of three."""
    instance = evaluator(name)
    stub(instance, native(name, level))
    result = instance(**INPUTS[name])
    assert result[name] == level + 1
    assert result[f"{name}_passed"] is passed
    assert result[f"{name}_result"] == ("pass" if passed else "fail")


@pytest.mark.parametrize("name", LIKERT_NAMES)
def test_tied_ratings_choose_lower_level_and_honor_custom_threshold(name):
    """Equal top probabilities select the lower level, which a custom threshold then judges."""
    instance = evaluator(name, threshold=4)
    stub(instance, native(name, probabilities={"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}))
    result = instance(**INPUTS[name])
    assert result[name] == 3
    assert result[f"{name}_passed"] is False


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("status", [0.499, 0.5])
def test_status_boundary_is_separate_from_score_and_retains_usage(name, status):
    """The applicability cutoff is applied before the score is read and a paid skip keeps its usage."""
    instance = evaluator(name)
    payload = native(name, status=status)
    if status < 0.5:
        payload["answers"][name] = {"type": "score", "probabilities": "unusable"}
    post = stub(instance, payload)
    result = instance(**INPUTS[name])
    post.assert_awaited_once()
    assert result[f"{name}_status"] == ("skipped" if status < 0.5 else "completed")
    assert result[f"{name}_total_tokens"] == 34
    if status < 0.5:
        assert result[name] is None
        assert result[f"{name}_passed"] is None
        assert result[f"{name}_result"] == "not_applicable"
        if name == "deflection_rate":
            assert result["deflection_rate_deflection_type"] == ""


@pytest.mark.parametrize("name", NAMES)
def test_malformed_native_score_fails_with_known_usage_and_no_chat_fallback(name):
    """An invalid native answer raises with the measured usage attached instead of falling back to chat."""
    instance = evaluator(name)
    payload = native(name)
    payload["answers"][name]["probabilities"] = {"0": 1.1}
    post = stub(instance, payload)
    with pytest.raises(EvaluationException) as error:
        instance(**INPUTS[name])
    post.assert_awaited_once()
    assert error.value.properties["total_tokens"] == 34


@pytest.mark.parametrize("name", NAMES)
def test_missing_required_evidence_is_rejected_before_transport(name):
    """Missing required evidence is rejected by validation before any native request is sent."""
    instance = evaluator(name)
    post = stub(instance, side_effect=AssertionError("No request expected"))
    inputs = deepcopy(INPUTS[name])
    inputs.pop("context" if name == "retrieval" else "response")
    with pytest.raises(EvaluationException):
        instance(**inputs)
    post.assert_not_called()


@pytest.mark.parametrize("name", [item for item in NAMES if item != "retrieval"])
def test_intermediate_tool_response_skips_without_transport(name):
    """An intermediate tool-call response is skipped deterministically without a paid request."""
    instance = evaluator(name)
    post = stub(instance, side_effect=AssertionError("No request expected"))
    inputs = deepcopy(INPUTS[name])
    inputs["response"] = [{"role": "assistant", "content": [
        {"type": "function_call", "name": "cancel_order", "arguments": {}, "tool_call_id": "call-1"},
    ]}]
    result = instance(**inputs)
    assert result[name] is None
    assert result[f"{name}_status"] == "skipped"
    post.assert_not_called()


@pytest.mark.parametrize("name", NAMES)
def test_paid_chat_skip_retains_usage_and_properties(name):
    """The unchanged chat parser retains known usage when a paid response declares inapplicability."""
    instance = evaluator(name, decision=False)
    instance._flow = AsyncMock(return_value={
        "llm_output": {"status": "skipped", "reason": "No usable evidence", "score": None,
                       "properties": {"diagnostic": "empty"}},
        "input_token_count": 11, "output_token_count": 3, "total_token_count": 14, "model_id": "chat-model",
    })
    result = instance(**INPUTS[name])
    instance._flow.assert_awaited_once()
    assert result[name] is None
    assert result[f"{name}_status"] == "skipped"
    assert result[f"{name}_total_tokens"] == 14
    assert result[f"{name}_properties"]["diagnostic"] == "empty"
    assert result[f"{name}_properties"]["total_tokens"] == 14


@pytest.mark.parametrize("name", NAMES)
def test_chat_transport_and_preparation_remain_unchanged(name):
    """The chat flow still receives the prepared inputs, plus only None-valued routing keys, and renders them."""
    instance = evaluator(name, decision=False)
    flow = instance._flow
    rendered = []

    async def render_only(timeout, **prepared):
        rendered.append(json.dumps(flow.render(**prepared)))
        return {"llm_output": {"score": 0 if name == "deflection_rate" else 4, "reason": "Mocked judgment."}}

    instance._flow = AsyncMock(side_effect=render_only)
    result = instance(**INPUTS[name])
    assert result[name] == (0 if name == "deflection_rate" else 4)
    kwargs = dict(instance._flow.call_args.kwargs)
    assert kwargs.pop("timeout") == instance._LLM_CALL_TIMEOUT
    assert {key: value for key, value in kwargs.items() if value is not None} == INPUTS[name]
    assert all(value is None for key, value in kwargs.items() if key not in INPUTS[name])
    for value in INPUTS[name].values():
        assert value in rendered[0]


@pytest.mark.parametrize(
    "level,type_name", tuple(enumerate(("plain_answer", "plain_denial", "send_elsewhere", "reframe")))
)
@pytest.mark.parametrize("threshold", [0, 1])
def test_deflection_categories_keep_binary_direction_and_type(level, type_name, threshold):
    """Each native category maps to the lower-is-better binary score and surfaces as the deflection type."""
    instance = evaluator("deflection_rate", threshold=threshold)
    stub(instance, native("deflection_rate", level))
    result = instance(**INPUTS["deflection_rate"])
    assert result["deflection_rate"] == int(level != 0)
    assert result["deflection_rate_passed"] is (int(level != 0) <= threshold)
    assert result["deflection_rate_deflection_type"] == type_name
    details = result["deflection_rate_properties"]["decision_model"]
    assert details["deflection_type"] == type_name
    assert details["native_category"] == level
    assert details["expected_score"] == int(level != 0)


def test_deflection_category_ties_and_concurrent_results_are_independent():
    """Tied categories select the lower one and concurrent evaluations never share state."""
    instance = evaluator("deflection_rate")

    async def transport(content, **kwargs):
        response = json.loads(content)["state"]["response"]
        await asyncio.sleep(0)
        if response == "tie":
            return native("deflection_rate", probabilities={"0": 0.5, "1": 0.5, "2": 0, "3": 0})
        return native("deflection_rate", 3)

    stub(instance, side_effect=transport)

    async def evaluate_both():
        return await asyncio.gather(instance._to_async()(response="tie"), instance._to_async()(response="other"))

    first, second = asyncio.run(evaluate_both())
    assert (first["deflection_rate"], first["deflection_rate_deflection_type"]) == (0, "plain_answer")
    assert (second["deflection_rate"], second["deflection_rate_deflection_type"]) == (1, "reframe")


@pytest.mark.parametrize("name", ["customer_satisfaction", "deflection_rate"])
@pytest.mark.parametrize("level", [None, "conversation", "turn"])
def test_conversation_levels_preserve_transcript_and_select_distinct_questions(name, level):
    """Each evaluation level sends exactly the fields its prompt renders and selects its own rubric."""
    instance = evaluator(name, evaluation_level=level)
    post = stub(instance, native(name))
    messages = deepcopy(MESSAGES)
    instance(messages=messages)
    state, questions = request(post)
    assert messages == MESSAGES
    if level == "turn":
        assert set(state) == ({"response"} if name == "deflection_rate" else {"query", "response"})
        assert "Your order is cancelled" in json.dumps(state["response"])
    else:
        assert set(state) == ({"messages"} if name == "deflection_rate" else {"conversation"})
        text = next(iter(state.values()))
        assert text.index("I cannot help") < text.index("That did not help") < text.index("Your order is cancelled")
        assert "later direct resolution" in questions[name]["instructions"] if name == "deflection_rate" else (
            "meaningful recovery" in questions[name]["instructions"]
        )


@pytest.mark.parametrize("name", ["relevance", "response_completeness", "similarity"])
def test_messages_preserve_reference_inputs_and_latest_turn_shape(name):
    """Messages select the latest turn and the native state keeps only the fields the prompt renders."""
    instance = evaluator(name)
    post = stub(instance, native(name))
    extra = {"ground_truth": "Order cancelled"} if name != "relevance" else {}
    instance(messages=deepcopy(MESSAGES), **extra)
    state, _ = request(post)
    assert set(state) == LATEST_TURN_STATE_KEYS[name]
    assert "Your order is cancelled" in json.dumps(state["response"])
    if extra:
        assert state["ground_truth"] == "Order cancelled"


def retrieval_messages():
    """Build a two-turn conversation whose tool outputs are the retrieval evidence."""
    return [
        {"role": "user", "content": "First query"},
        {"role": "tool", "content": "First context"},
        {"role": "assistant", "content": "First answer is not retrieval evidence"},
        {"role": "user", "content": "Second query"},
        {"role": "tool", "content": [{"type": "tool_result", "tool_result": {"city": "Paris"}}]},
        {"role": "assistant", "content": "Second answer is not retrieval evidence"},
    ]


def test_retrieval_extracts_ordered_tool_context_and_sums_split_usage():
    """Retrieval evaluates each user turn with its tool context and sums the usage across the turns."""
    instance = evaluator("retrieval")
    post = stub(instance, side_effect=[native("retrieval", 1), native("retrieval", 4)])
    result = instance(messages=retrieval_messages())
    assert post.await_count == 2
    assert request(post, 0)[0] == {"query": "First query", "context": "First context"}
    assert request(post, 1)[0] == {"query": "Second query", "context": '{"city": "Paris"}'}
    assert result["retrieval"] == 3.5
    assert result["retrieval_total_tokens"] == 68
    assert result["retrieval_model"] == "actual-model"


@pytest.mark.parametrize(
    "name", ["customer_satisfaction", "relevance", "response_completeness", "similarity", "retrieval"]
)
@pytest.mark.parametrize("fail_second", [False, True])
def test_split_turn_usage_survives_aggregation_and_later_failure(name, fail_second):
    """Split conversations sum paid usage on success and keep it on the error when a later turn fails."""
    instance = evaluator(name)
    second = native(name, 4)
    if fail_second:
        second["answers"][name]["probabilities"] = {"0": -1}
    post = stub(instance, side_effect=[native(name, 0), second])
    if name == "retrieval":
        inputs = {"messages": retrieval_messages()}
    else:
        messages = deepcopy(MESSAGES)
        for message in messages:
            if message["role"] == "assistant":
                message["ground_truth"] = "Order cancelled"
        inputs = {"conversation": {"messages": messages}}
    if fail_second:
        with pytest.raises(EvaluationException) as error:
            instance(**inputs)
        assert error.value.properties["total_tokens"] == 68
        assert error.value.properties["decision_model"]["result_usage_scope"] == "reported_turns_only"
    else:
        result = instance(**inputs)
        assert result[f"{name}_total_tokens"] == 68
        assert result[name] == 3
    assert post.await_count == 2


@pytest.mark.parametrize("name", NAMES)
def test_unsupported_sdk_rejects_native_before_loading_prompty(name):
    """Without Decision support the native configuration fails clearly and no chat prompt is loaded."""
    with patch.object(module(name), "_SDK_SUPPORTS_DECISION", False), patch(
        PROMPTY_LOAD, side_effect=AssertionError("Chat must not initialize")
    ):
        with pytest.raises(EvaluationException, match="version with decision model support"):
            evaluator(name)


@pytest.mark.parametrize("name", NAMES)
def test_rubrics_are_contained_in_each_independent_asset(name):
    """Each asset package ships exactly the rubric files its evaluation levels need."""
    path = Path(module(name).__file__).parent
    expected = {f"{name}.decision.yaml"}
    if name in ("customer_satisfaction", "deflection_rate"):
        expected.add(f"{name}_multi_turn.decision.yaml")
    assert {item.name for item in path.glob("*.decision.yaml")} == expected
