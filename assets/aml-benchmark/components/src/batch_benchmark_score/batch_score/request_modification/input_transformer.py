# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for input transformer."""

from typing import Any
from .modifiers.request_modifier import RequestModifier
from ..utils import logging_utils as lu


class InputTransformer():
    """Class for input transformer."""

    def __init__(self, modifiers: "list[RequestModifier]") -> None:
        """Init class."""
        self.__modifiers = modifiers

    def apply_modifications(self, request_obj: Any) -> Any:
        """Apply modifications."""
        if self.__modifiers and len(self.__modifiers) > 0:
            lu.get_logger().info("Applying InputTransformer modifications")
            for modifier in self.__modifiers:
                request_obj = modifier.modify(request_obj)
        else:
            lu.get_logger().info("No InputTransformer modifications configured")

        return request_obj
