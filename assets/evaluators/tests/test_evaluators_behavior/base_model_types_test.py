# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared coverage tests for evaluator model type selection and SDK compatibility."""

import asyncio
import builtins
import runpy
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from azure.ai.evaluation._evaluators import _common

from ..test_decision_models.test_released_sdk_compatibility import (
    BUILTIN,
    CHAT_CONFIG,
    DECISION_CONFIG,
    DECISION_MODULE,
    MESSAGES,
    STRATEGY_MODULE,
    _exercise_chat,
    _inputs,
)


class ModelTypesTests:
    """Exercise model type contracts without requiring a Decision-capable installed SDK."""

    def _load_module(self):
        return runpy.run_path(str(BUILTIN / self.result_key / "evaluator" / f"_{self.result_key}.py"))

    def test_legacy_chat_model_type(self, monkeypatch):
        """Keep sync and async chat scoring functional when shared SDK support is absent."""
        monkeypatch.delattr(_common, "ModelEvaluatorBase", raising=False)
        monkeypatch.setitem(sys.modules, STRATEGY_MODULE, None)
        monkeypatch.setitem(sys.modules, DECISION_MODULE, None)
        module = self._load_module()
        assert module["PromptyEvaluationStrategy"] is None
        assert not module["_SDK_SUPPORTS_DECISION"]
        _exercise_chat(module, self.result_key, self.class_name)

    @pytest.mark.parametrize("module_name", [STRATEGY_MODULE, DECISION_MODULE])
    def test_broken_sdk_import_is_not_hidden(self, monkeypatch, module_name):
        """Propagate SDK dependency failures rather than treating them as absent capabilities."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == module_name:
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            self._load_module()
        assert caught.value is error

    def test_decision_model_type_routing(self, monkeypatch):
        """Load native rubrics and delegate prepared inputs to the selected strategy."""
        module = self._load_module()
        evaluator_class = module[self.class_name]
        namespace = module["_load_decision_questions"].__globals__
        base = module["ModelEvaluatorBase"]
        original_init = base.__init__
        runner = Mock(probability_threshold=0.8)

        def initialize_chat(instance, **kwargs):
            kwargs["model_config"] = dict(CHAT_CONFIG)
            original_init(instance, **kwargs)

        monkeypatch.setattr(base, "__init__", initialize_chat)
        monkeypatch.setattr(base, "_decision_model_runner", property(lambda instance: runner), raising=False)
        monkeypatch.setitem(namespace, "_SDK_SUPPORTS_DECISION", True)
        scoring = Mock()
        monkeypatch.setitem(namespace, "BinaryScoring" if self.result_key == "task_completion" else "LikertScoring",
                            scoring)
        strategy_factory = Mock(side_effect=lambda *args, **kwargs: Mock(
            evaluate=AsyncMock(return_value={self.result_key: 1})
        ))
        monkeypatch.setitem(namespace, "DecisionEvaluationStrategy", strategy_factory)
        with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", return_value=AsyncMock()):
            evaluator = evaluator_class(dict(DECISION_CONFIG))
        expected_count = 2 if self.result_key == "task_completion" else 1
        assert strategy_factory.call_count == expected_count
        for call in strategy_factory.call_args_list:
            assert call.args == (runner,)
            assert set(call.kwargs["questions"]) == {"status", self.result_key}
            assert call.kwargs["result_key"] == self.result_key
            assert call.kwargs["scoring"] is scoring.return_value
        result = asyncio.run(evaluator._do_eval(_inputs(self.result_key)))
        assert result == {self.result_key: 1}
        evaluator._strategy.evaluate.assert_awaited_once()
        if self.result_key == "task_completion":
            assert scoring.call_args.args == (0.8,)
            result = asyncio.run(evaluator._do_eval({"messages": MESSAGES}))
            assert result == {self.result_key: 1}
            evaluator._multi_turn_strategy.evaluate.assert_awaited_once()
            prepared = evaluator._multi_turn_strategy.evaluate.await_args.args[0]
            assert "User turn 1:" in prepared["messages"]
        else:
            scoring.assert_called_once_with(5, minimum=1)
