# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for admin-connected (BYO) judge model support in the built-in prompty evaluators.

Admin-connected (BYO) models are referenced as ``"connection-name/deployment-name"`` and are only
invokable through the Foundry project **Responses API** (the platform resolves the connection and
every auth type). A BYO model configuration carries ``byo_model`` + ``project_endpoint`` and omits
``azure_endpoint`` / ``azure_deployment``.

The built-in prompty judges (coherence, relevance, fluency, groundedness, ...) are thin
pass-throughs over ``azure-ai-evaluation`` — all BYO routing lives in the SDK. The per-evaluator
guard that each shipped evaluator accepts and forwards a BYO config lives in
``ByoJudgeBehaviorMixin`` (mixed into ``BaseEvaluatorBehaviorTest``), so it runs automatically
whenever an evaluator's code changes. This module adds two cross-cutting guards: the SDK
BYO-detection contract the evaluators rely on, and an end-to-end judge call routed through the BYO
project-Responses client (network mocked) for a representative set of core judges.
"""

import json

import pytest
from unittest.mock import AsyncMock, patch

from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator
from ...builtin.fluency.evaluator._fluency import FluencyEvaluator
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator

from .byo_judge_behavior_mixin import BYO_MODEL, PROJECT_ENDPOINT, is_byo_model_config
from .byo_judge_behavior_mixin import BYO_CLIENT_PATH as _BYO_CLIENT_PATH
from .byo_judge_behavior_mixin import byo_config as _byo_config
from .byo_judge_behavior_mixin import make_chat_completion as _make_chat_completion


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
