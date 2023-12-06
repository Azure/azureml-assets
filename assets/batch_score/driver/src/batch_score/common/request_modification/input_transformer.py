# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .modifiers.request_modifier import RequestModifier


class InputTransformer():
    def __init__(self, modifiers: "list[RequestModifier]") -> None:
        self.__modifiers = modifiers or []
    
    def apply_modifications(self, request_obj: any):
        for modifier in self.__modifiers:
            request_obj = modifier.modify(request_obj)

        return request_obj
