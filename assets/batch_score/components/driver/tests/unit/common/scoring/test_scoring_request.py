# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for scoring request."""

from src.batch_score.root.common.scoring.scoring_request import ScoringRequest
from src.batch_score.root.common.request_modification.modifiers.input_type_modifier import InputTypeModifier
from src.batch_score.root.common.request_modification.input_transformer import InputTransformer
from src.batch_score.root.common.common_enums import InputType


def test_extract_input_type():
    """Test extraction of input type."""
    input_type_modifier = InputTypeModifier()
    input_transfomer = InputTransformer([input_type_modifier])
    result = ScoringRequest(
        original_payload='{"messages": [{"content": "just text"}]}',
        input_to_request_transformer=input_transfomer,
    )

    assert result.input_type == InputType.TextOnly


def test_extract_input_type_unknown():
    """Test extraction of input type when it cannot be determined."""
    input_type_modifier = InputTypeModifier()
    input_transfomer = InputTransformer([input_type_modifier])
    result = ScoringRequest(
        original_payload='{"messages": []}',
        input_to_request_transformer=input_transfomer,
    )

    assert result.input_type == InputType.Unknown


def test_extract_input_type_default():
    """Test extraction of input type when there are no transformers."""
    result = ScoringRequest(
        original_payload='{"messages": []}',
    )

    assert result.input_type == InputType.Unknown
