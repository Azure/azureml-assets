# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock input transformer."""

import pytest

from src.batch_score.common.request_modification.input_transformer import (
    InputTransformer,
)
from src.batch_score.common.request_modification.modifiers.request_modifier import (
    RequestModifier,
)


@pytest.fixture
def make_input_transformer():
    """Make a mock input transformer."""
    def make(modifiers: "list[RequestModifier]" = None):
        return InputTransformer(
            modifiers=modifiers
        )

    return make


class FakeRequestModifier(RequestModifier):
    """Mock request modifier."""

    def __init__(self, prefix: str = "modified") -> None:
        """Mock init function."""
        self.__prefix = prefix
        self.__counter: int = 0

    def modify(self, request_obj: any) -> any:
        """Mock modify function."""
        self.__counter = self.__counter + 1
        return {f"{self.__prefix}": f"{self.__counter}"}


class FakeInputOutputModifier(RequestModifier):
    """Mock input output modifier."""

    CHANGED_OUTPUT = "<CHANGED OUTPUT>"

    def modify(self, request_obj: any) -> any:
        """Mock modify function."""
        for key in request_obj:
            request_obj[key] = FakeInputOutputModifier.CHANGED_OUTPUT

        return request_obj
