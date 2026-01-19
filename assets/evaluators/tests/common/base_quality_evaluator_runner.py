# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for quality tests of evaluators with real flow execution (no mocking).
"""

from .base_evaluator_runner import BaseEvaluatorRunner


class BaseQualityEvaluatorRunner(BaseEvaluatorRunner):
    """
    Base class for quality tests that use real LLM flow execution.

    This is a thin wrapper around BaseEvaluatorRunner that disables mocking.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator
    """

    use_mocking = False  # Quality tests always use real flow execution
