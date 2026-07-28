# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for admin-connected (BYO) judge model support in the built-in prompty evaluators.

Admin-connected (BYO) models are referenced as ``"connection-name/deployment-name"`` and are only
invokable through the Foundry project **Responses API** (the platform resolves the connection and
every auth type). A BYO model configuration carries ``byo_model`` + ``project_endpoint`` and omits
``azure_endpoint`` / ``azure_deployment``.

The built-in prompty judges (coherence, relevance, fluency, groundedness, ...) are thin
pass-throughs over ``azure-ai-evaluation`` — all BYO routing lives in the SDK. These tests guard the
asset's contract: that the shipped evaluators (1) accept a BYO config and forward it unchanged so the
SDK recognises it as BYO, and (2) end-to-end route a judge call through the BYO project-Responses
client (with the network call mocked), rather than a direct Azure OpenAI client.
"""

import json

import pytest
from unittest.mock import AsyncMock, patch

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from azure.ai.evaluation._byo_judge import is_byo_model_config

from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator
from ...builtin.fluency.evaluator._fluency import FluencyEvaluator
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator


BYO_MODEL = "my-conn/gpt-4o-mini"
PROJECT_ENDPOINT = "https://acct.services.ai.azure.com/api/projects/proj"

# The SDK prompty imports the BYO client into its own module namespace; patch it there so the
# built-in evaluator's judge call is intercepted before any network I/O.
_BYO_CLIENT_PATH = "azure.ai.evaluation._legacy.prompty._prompty.AsyncByoProjectResponsesClient"


def _byo_config(byo_model=BYO_MODEL, project_endpoint=PROJECT_ENDPOINT):
    """Build a minimal admin-connected (BYO) model configuration (both markers, no AOAI fields)."""
    return {"byo_model": byo_model, "project_endpoint": project_endpoint}


def _make_chat_completion(content):
    """Build a ``ChatCompletion`` carrying the judge's raw JSON output (the shim's return shape)."""
    return ChatCompletion(
        id="byo-test",
        created=0,
        model="byo-model",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
    )


# (EvaluatorClass, metric_name, call_kwargs) for the core grader-style prompty judges (1-5 scale).
_CORE_JUDGES = [
    (
        CoherenceEvaluator,
        "coherence",
        {"query": "How does a fridge keep food cold?",
         "response": "It moves heat out of the cabinet using a refrigerant cycle."},
    ),
    (
        RelevanceEvaluator,
        "relevance",
        {"query": "How does a fridge keep food cold?",
         "response": "It moves heat out of the cabinet using a refrigerant cycle."},
    ),
    (
        FluencyEvaluator,
        "fluency",
        {"response": "It moves heat out of the cabinet using a refrigerant cycle."},
    ),
    (
        GroundednessEvaluator,
        "groundedness",
        {"query": "How does a fridge keep food cold?",
         "context": "A refrigerator uses a refrigerant cycle to move heat from inside the cabinet to outside.",
         "response": "It moves heat out of the cabinet using a refrigerant cycle."},
    ),
]

_CORE_IDS = [name for _, name, _ in _CORE_JUDGES]


@pytest.mark.unittest
class TestIsByoModelConfigContract:
    """The BYO-detection contract (``azure-ai-evaluation``) that the built-in evaluators rely on."""

    def test_true_when_both_markers_present(self):
        """Detect a BYO config when both markers are present."""
        assert is_byo_model_config(_byo_config()) is True

    def test_false_for_azure_openai_config(self):
        """Reject a direct Azure OpenAI config as non-BYO."""
        assert is_byo_model_config(
            {"azure_endpoint": "https://x.openai.azure.com", "azure_deployment": "gpt-4o-mini"}
        ) is False

    def test_false_when_only_byo_model(self):
        """Reject a config carrying only ``byo_model`` as non-BYO."""
        assert is_byo_model_config({"byo_model": BYO_MODEL}) is False

    def test_false_when_only_project_endpoint(self):
        """Reject a config carrying only ``project_endpoint`` as non-BYO."""
        assert is_byo_model_config({"project_endpoint": PROJECT_ENDPOINT}) is False

    def test_false_for_empty_or_none(self):
        """Reject an empty or ``None`` config as non-BYO."""
        assert is_byo_model_config({}) is False
        assert is_byo_model_config(None) is False

    def test_false_for_non_string_markers(self):
        """Reject non-string BYO markers as non-BYO."""
        assert is_byo_model_config({"byo_model": 1, "project_endpoint": 2}) is False


@pytest.mark.unittest
@pytest.mark.parametrize("evaluator_cls, name, _call_kwargs", _CORE_JUDGES, ids=_CORE_IDS)
class TestByoConfigForwarding:
    """Each shipped prompty judge accepts a BYO config and forwards it intact to the SDK prompty."""

    def test_byo_config_forwarded_and_detected_as_byo(self, evaluator_cls, name, _call_kwargs):
        """The evaluator forwards the BYO markers so the SDK routes chat.completions to Responses."""
        evaluator = evaluator_cls(model_config=_byo_config())
        configuration = evaluator._flow._model.configuration
        assert is_byo_model_config(configuration) is True
        assert configuration["byo_model"] == BYO_MODEL
        assert configuration["project_endpoint"] == PROJECT_ENDPOINT

    def test_byo_config_omits_azure_openai_markers(self, evaluator_cls, name, _call_kwargs):
        """A BYO config is accepted without the ``azure_endpoint`` / ``azure_deployment`` fields."""
        evaluator = evaluator_cls(model_config=_byo_config())
        configuration = evaluator._flow._model.configuration
        assert "azure_endpoint" not in configuration
        assert "azure_deployment" not in configuration


@pytest.mark.unittest
@pytest.mark.parametrize("evaluator_cls, name, call_kwargs", _CORE_JUDGES, ids=_CORE_IDS)
class TestByoRoutingEndToEnd:
    """End-to-end: a BYO judge call routes through the project Responses client (network mocked)."""

    def test_byo_judge_call_routes_through_project_responses(self, evaluator_cls, name, call_kwargs):
        """Route a BYO judge call through the project Responses client, not a direct client."""
        judge_output = json.dumps({"score": 4, "reason": "Grounded, coherent and relevant."})
        with patch(_BYO_CLIENT_PATH) as mock_byo_client_cls:
            client = mock_byo_client_cls.return_value
            client.with_options.return_value = client
            client.chat.completions.create = AsyncMock(return_value=_make_chat_completion(judge_output))

            evaluator = evaluator_cls(model_config=_byo_config())
            result = evaluator(**call_kwargs)

        # Routing: the BYO project-Responses client was built from the connection/deployment + endpoint.
        mock_byo_client_cls.assert_called_once()
        ctor_kwargs = mock_byo_client_cls.call_args.kwargs
        assert ctor_kwargs["byo_model"] == BYO_MODEL
        assert ctor_kwargs["project_endpoint"] == PROJECT_ENDPOINT
        # The judge call went through the BYO client (not a direct AzureOpenAI/OpenAI client).
        client.chat.completions.create.assert_awaited()
        # The evaluator parsed the mocked judge output into a passing score (grader threshold is 3).
        assert result[name] == 4.0
        assert result[f"{name}_score"] == 4.0
        assert result[f"{name}_result"] == "pass"
