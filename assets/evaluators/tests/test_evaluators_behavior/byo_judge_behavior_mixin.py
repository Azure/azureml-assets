# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable admin-connected (BYO) judge behavior guard for prompty evaluators.

Admin-connected (BYO) models are referenced as ``"connection-name/deployment-name"`` and are only
invokable through the Foundry project **Responses API** (the platform resolves the connection and
every auth type). A BYO model configuration carries ``byo_model`` + ``project_endpoint`` and omits
``azure_endpoint`` / ``azure_deployment``.

The built-in prompty judges are thin pass-throughs over ``azure-ai-evaluation`` — all BYO routing
lives in the SDK. ``ByoJudgeBehaviorMixin`` is mixed into the shared prompty behavior base
(``BaseEvaluatorBehaviorTest``) so every built-in prompty evaluator's behavior suite runs this guard
whenever that evaluator's code changes: the evaluator must accept a BYO ``model_config`` and forward
it unchanged so the SDK recognises it as BYO and routes the judge call through the project-Responses
client rather than a direct Azure OpenAI client.
"""

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from azure.ai.evaluation._byo_judge import is_byo_model_config

BYO_MODEL = "my-conn/gpt-4o-mini"
PROJECT_ENDPOINT = "https://acct.services.ai.azure.com/api/projects/proj"

# The SDK prompty imports the BYO client into its own module namespace; patch it there so a
# built-in evaluator's judge call is intercepted before any network I/O.
BYO_CLIENT_PATH = "azure.ai.evaluation._legacy.prompty._prompty.AsyncByoProjectResponsesClient"


def byo_config(byo_model=BYO_MODEL, project_endpoint=PROJECT_ENDPOINT):
    """Build a minimal admin-connected (BYO) model configuration (both markers, no AOAI fields)."""
    return {"byo_model": byo_model, "project_endpoint": project_endpoint}


def make_chat_completion(content):
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


class ByoJudgeBehaviorMixin:
    """Per-evaluator admin-connected (BYO) judge guard, mixed into each prompty behavior suite.

    Runs against ``self.evaluator_type`` (set by the concrete evaluator behavior class), so no
    per-evaluator input data is needed: constructing the evaluator with a BYO ``model_config`` and
    inspecting the forwarded flow configuration is input-independent and works for every prompty
    evaluator. End-to-end judge-call routing (which is input/output-schema specific per evaluator)
    is covered separately for representative judges in ``test_byo_judge_behavior.py``.
    """

    def test_accepts_and_forwards_byo_model_config(self):
        """Accept a BYO model_config and forward it so the SDK detects BYO and routes via Responses."""
        evaluator = self.evaluator_type(model_config=byo_config())

        configuration = evaluator._flow._model.configuration
        assert is_byo_model_config(configuration) is True
        assert configuration["byo_model"] == BYO_MODEL
        assert configuration["project_endpoint"] == PROJECT_ENDPOINT
        # A BYO config must not carry the direct Azure OpenAI markers, or the SDK would take the
        # direct client path instead of the project-Responses path.
        assert "azure_endpoint" not in configuration
        assert "azure_deployment" not in configuration
