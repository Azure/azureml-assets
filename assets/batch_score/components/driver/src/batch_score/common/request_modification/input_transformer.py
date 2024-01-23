# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Input transformer."""

from .modifiers.request_modifier import RequestModifier


class InputTransformer():
    """Input transformer."""

    def __init__(self, modifiers: "list[RequestModifier]") -> None:
        """Initialize InputTransformer."""
        self.__modifiers = modifiers or []

    def apply_modifications(self, request_obj: any):
        """Apply modifications."""
        for modifier in self.__modifiers:
            request_obj = modifier.modify(request_obj)

        return request_obj
