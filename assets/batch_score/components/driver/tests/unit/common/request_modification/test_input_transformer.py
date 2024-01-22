# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for input transformer."""

from src.batch_score.common.request_modification.input_transformer import (
    InputTransformer,
)
from tests.fixtures.input_transformer import FakeRequestModifier


def test_input_transformer(mock_get_logger, make_input_transformer):
    input_transformer: InputTransformer = make_input_transformer(modifiers=[FakeRequestModifier(prefix="first"), FakeRequestModifier(prefix="second")])
    
    modified_obj = input_transformer.apply_modifications({"mock": "value"})

    assert modified_obj == {"second": "1"}

def test_empty_input_transformer(mock_get_logger, make_input_transformer):
    input_transformer: InputTransformer = make_input_transformer(modifiers=None)
    modified_obj = input_transformer.apply_modifications({"mock": "value"})
    assert modified_obj["mock"] == "value"
    
    input_transformer: InputTransformer = make_input_transformer(modifiers=[])
    modified_obj = input_transformer.apply_modifications({"mock": "value"})
    assert modified_obj["mock"] == "value"