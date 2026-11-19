# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Exercise missing-evidence applicability through the real asset entry points."""
import json
from unittest.mock import AsyncMock

import pytest
from azure.ai.evaluation._common.utils import serialize_messages
from azure.ai.evaluation._exceptions import EvaluationException
from ...builtin.output_quality.evaluator._output_quality import OutputQualityEvaluator
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallAccuracyEvaluator
from .test_composites import CONFIG, CHAT_CONFIG, _native


class TestInputApplicability:
    """Scorers receive existing evidence; absent calls yield an unscored skip."""

    def test_output_quality_without_user_scores_applicable_response_members(self):
        evaluator = OutputQualityEvaluator(CONFIG, evaluation_level="conversation")
        messages = [{"role": "assistant", "content": "Hello. I can help with that."}]
        payload = _native(OutputQualityEvaluator, skipped=True)
        payload["answers"]["fluency_status"]["noul"] = 1
        post = AsyncMock(return_value=payload)
        evaluator._decision_model_runner._post = post
        if not hasattr(type(evaluator._validator), "requires_assistant"):
            with pytest.raises(EvaluationException, match="role 'user'"):
                evaluator(messages=messages)
            post.assert_not_called()
            return
        result = evaluator(messages=messages)
        assert result["fluency_status"] == "completed"
        assert result["fluency_score"] == 5
        assert result["task_completion_status"] == "skipped"
        assert json.loads(post.call_args.args[0])["state"]["messages"] == serialize_messages(messages)

    @pytest.mark.parametrize("evaluator_class", [TaskCompletionEvaluator, OutputQualityEvaluator])
    def test_user_goal_without_assistant_reaches_judge_as_original_transcript(self, evaluator_class):
        evaluator = evaluator_class(CONFIG, evaluation_level="conversation")
        messages = [{"role": "user", "content": "Book a flight to Paris."}]
        if evaluator_class is OutputQualityEvaluator:
            payload = _native(OutputQualityEvaluator, skipped=True)
            payload["answers"]["task_completion_status"]["noul"] = 1
        else:
            payload = {"model": "actual-model", "usage": {"input_tokens": 30, "output_tokens": 4},
                       "answers": {"status": {"type": "noul", "noul": 1}}}
        payload["answers"]["task_completion"] = {"type": "noul", "noul": .1}
        post = AsyncMock(return_value=payload)
        evaluator._decision_model_runner._post = post
        if not hasattr(type(evaluator._validator), "requires_assistant"):
            with pytest.raises(EvaluationException, match="role 'assistant'"):
                evaluator(messages=messages)
            post.assert_not_called()
            return
        result = evaluator(messages=messages)
        assert result["task_completion_status"] == "completed"
        assert result["task_completion_result"] == "fail"
        assert json.loads(post.call_args.args[0])["state"]["messages"] == serialize_messages(messages)

    @pytest.mark.parametrize("config", [CONFIG, CHAT_CONFIG], ids=["decision", "chat"])
    @pytest.mark.parametrize("response", [None, [], ""])
    def test_absent_calls_skip_without_judge_request(self, config, response):
        evaluator = ToolCallAccuracyEvaluator(config)
        evaluator._do_eval = AsyncMock(side_effect=AssertionError("No calls to judge"))
        if not hasattr(type(evaluator._validator), "allow_empty_calls"):
            with pytest.raises(EvaluationException, match="No tool calls"):
                evaluator(query="Find a flight.", response=response,
                          tool_definitions=[{"name": "search", "parameters": {}}])
            evaluator._do_eval.assert_not_called()
            return
        result = evaluator(query="Find a flight.", response=response,
                           tool_definitions=[{"name": "search", "parameters": {}}])
        assert result["tool_call_accuracy_status"] == "skipped"
        assert result["tool_call_accuracy_score"] is None
        assert result["tool_call_accuracy_passed"] is None
        evaluator._do_eval.assert_not_called()

    @pytest.mark.parametrize("config", [CONFIG, CHAT_CONFIG], ids=["decision", "chat"])
    @pytest.mark.parametrize("changes", [{"query": None}, {"response": 123}, {"tool_calls": [123]}])
    def test_malformed_inputs_still_raise(self, config, changes):
        evaluator = ToolCallAccuracyEvaluator(config)
        inputs = {"query": "Find a flight.", "response": [],
                  "tool_definitions": [{"name": "search", "parameters": {}}]}
        inputs.update(changes)
        with pytest.raises(EvaluationException):
            evaluator(**inputs)
