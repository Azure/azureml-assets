# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Output Utilization Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import ToolOutputUtilizationEvaluator


@pytest.mark.quality
class TestToolOutputUtilizationEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Output Utilization Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolOutputUtilizationEvaluator

    # TODO: Add specific test cases for tool output utilization evaluation
    # Test cases should include:
    # - Responses that effectively utilize tool outputs
    # - Responses that partially utilize tool outputs
    # - Responses that ignore available tool outputs
    # - Edge cases (multiple tool outputs, conflicting tool outputs)
