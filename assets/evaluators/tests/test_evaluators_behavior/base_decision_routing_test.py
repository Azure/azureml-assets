# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helpers for covering native Decision routing on the released SDK the asset tests install.

The per-asset test job installs the released ``azure-ai-evaluation`` pinned in ``tests/requirements.txt``, which has
no Decision model support, so the native branches of an evaluator module are unreachable unless the module is made
to believe the SDK supports them. These helpers load a fresh copy of the module, flip its capability flag, replace
the strategy and scoring names with mocks and let construction with a Decision configuration proceed as chat
construction, so the routing code runs and can be asserted on without a Decision-capable SDK or any transport.
"""

import asyncio
import inspect
import runpy
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

BUILTIN = Path(__file__).parents[2] / "builtin"
CHAT_CONFIG = {
    "azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "chat", "api_key": "test-only",
}
DECISION_CONFIG = {**CHAT_CONFIG, "type": "decision", "azure_deployment": "Decision-1"}
PROMPTY_LOAD = "azure.ai.evaluation._legacy.prompty.AsyncPrompty.load"


def load_evaluator_module(name):
    """Load a fresh copy of ``builtin/<name>/evaluator/_<name>.py`` so patched globals never leak between tests."""
    return runpy.run_path(str(BUILTIN / name / "evaluator" / f"_{name}.py"))


def module_namespace(module, class_name=None):
    """Return the live globals of a loaded module.

    ``run_path`` hands back a copy of the namespace; the functions defined in the module keep the original.
    """
    for value in module.values():
        if isinstance(value, types.FunctionType) and value.__globals__.get("__file__") == module.get("__file__"):
            return value.__globals__
    # Decorated classes (for example ``@experimental``) wrap ``__init__``; unwrap to reach the module's function.
    return inspect.unwrap(module[class_name].__init__).__globals__


class FakeDecisionSdk:
    """The mocks a module sees after :func:`enable_fake_decision_sdk`."""

    def __init__(self, factory, runner, scorings):
        """Keep the strategy factory, the runner stub and the scoring mocks by name."""
        self.factory = factory
        self.runner = runner
        self.scorings = scorings

    @property
    def strategies(self):
        """Every strategy the module constructed, in construction order."""
        return list(self.factory.side_effect_results)


def enable_fake_decision_sdk(monkeypatch, module, class_name, *, result_key, scorings=("LikertScoring",),
                             probability_threshold=0.8, evaluate=None):
    """Make a loaded module believe the installed SDK supports Decision models.

    The module keeps the released ``PromptyEvaluatorBase`` under its ``ModelEvaluatorBase`` alias. Its initializer
    is wrapped so a Decision configuration is constructed as chat, the runner property returns a stub carrying
    ``probability_threshold``, and the strategy and scoring names are replaced with mocks. ``evaluate`` is the
    value every strategy's ``evaluate`` returns (default ``{result_key: 1}``) or a callable used as side effect.
    """
    namespace = module_namespace(module, class_name)
    base = module["ModelEvaluatorBase"]
    original_init = base.__init__

    def initialize_as_chat(instance, **kwargs):
        kwargs["model_config"] = dict(CHAT_CONFIG)
        original_init(instance, **kwargs)

    runner = Mock(name="runner", probability_threshold=probability_threshold)
    monkeypatch.setattr(base, "__init__", initialize_as_chat)
    monkeypatch.setattr(base, "_decision_model_runner", property(lambda instance: runner), raising=False)
    monkeypatch.setitem(namespace, "_SDK_SUPPORTS_DECISION", True)
    scoring_mocks = {}
    for name in scorings:
        scoring_mocks[name] = Mock(name=name)
        monkeypatch.setitem(namespace, name, scoring_mocks[name])
    results = []

    def make_strategy(*args, **kwargs):
        strategy = Mock(name="strategy")
        if callable(evaluate):
            strategy.evaluate = AsyncMock(side_effect=evaluate)
        else:
            strategy.evaluate = AsyncMock(return_value={result_key: 1} if evaluate is None else evaluate)
        strategy.construction = (args, kwargs)
        results.append(strategy)
        return strategy

    factory = Mock(name="DecisionEvaluationStrategy", side_effect=make_strategy)
    factory.side_effect_results = results
    monkeypatch.setitem(namespace, "DecisionEvaluationStrategy", factory)
    return FakeDecisionSdk(factory, runner, scoring_mocks)


def construct_evaluator(module, class_name, config=None, **kwargs):
    """Construct an evaluator with Prompty loading replaced; returns the evaluator and the load mock."""
    with patch(PROMPTY_LOAD, return_value=AsyncMock()) as load:
        evaluator = module[class_name](dict(DECISION_CONFIG if config is None else config), **kwargs)
    return evaluator, load


def run(coroutine):
    """Run a coroutine to completion on a fresh loop."""
    return asyncio.run(coroutine)


class DecisionRoutingTests:
    """Routing coverage every expanded evaluator needs on the released SDK.

    Subclasses set ``result_key``, ``class_name``, ``inputs`` (turn-level inputs that reach the strategy) and, when
    they differ from the defaults, ``strategy_attribute`` (the attribute holding the turn-level strategy),
    ``scorings`` (the scoring names the module takes from the SDK) and ``expected_strategies`` (how many
    strategies construction creates). Module-specific branches (multi-turn, no-query, probability cutoffs,
    deterministic skips) belong in the subclass.
    """

    result_key: str
    class_name: str
    inputs: dict
    strategy_attribute = "_strategy"
    scorings = ("LikertScoring",)
    expected_strategies = 1

    def load(self):
        """Load a fresh module copy for this evaluator."""
        return load_evaluator_module(self.result_key)

    def test_unsupported_sdk_rejects_native_before_loading_prompty(self, monkeypatch):
        """Without Decision support the native configuration fails clearly and no prompt is loaded."""
        module = self.load()
        monkeypatch.setitem(module_namespace(module, self.class_name), "_SDK_SUPPORTS_DECISION", False)
        with patch(PROMPTY_LOAD) as load:
            with pytest.raises(EvaluationException, match="[Dd]ecision"):
                module[self.class_name](dict(DECISION_CONFIG))
        load.assert_not_called()

    def test_native_routing_builds_strategies_and_delegates_prepared_inputs(self, monkeypatch):
        """Native construction loads the packaged rubrics and ``_do_eval`` hands prepared inputs to the strategy."""
        module = self.load()
        sdk = enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                       scorings=self.scorings)
        evaluator, _ = construct_evaluator(module, self.class_name)
        assert sdk.factory.call_count == self.expected_strategies
        for args, kwargs in (strategy.construction for strategy in sdk.strategies):
            assert args == (sdk.runner,)
            assert set(kwargs["questions"]) == {"status", self.result_key}
            assert kwargs["result_key"] == self.result_key
        result = run(evaluator._do_eval(dict(self.inputs)))
        assert result == {self.result_key: 1}
        getattr(evaluator, self.strategy_attribute).evaluate.assert_awaited_once()
        state = getattr(evaluator, self.strategy_attribute).evaluate.await_args.args[0]
        assert None not in state.values()
