# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Input type modifier."""

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from ...common_enums import InputType
from ...constants import (
    IMAGE_CONTENT_TYPE,
    IMAGE_URL_CONTENT_TYPE,
    TEXT_CONTENT_TYPE
)


class InputTypeModifier(RequestModifier):
    """Identifies request by type input and adds the result to the request."""

    __INPUT_TYPE_PROPERTY = "input_type"

    @staticmethod
    def get_input_type(request_obj: any) -> InputType:
        """Classify the request based on message type."""
        inputType = InputType.Unknown
        if not isinstance(request_obj, dict) or "messages" not in request_obj or not request_obj["messages"]:
            return inputType

        for message in request_obj["messages"]:
            if "content" not in message or not message["content"]:
                continue
            for item in message["content"]:
                if isinstance(item, str):
                    if inputType == InputType.Unknown:
                        inputType = InputType.TextOnly
                    elif inputType == InputType.Image:
                        inputType = InputType.ImageAndText
                elif isinstance(item, dict) and "type" in item:
                    if item["type"] == IMAGE_CONTENT_TYPE or item["type"] == IMAGE_URL_CONTENT_TYPE:
                        if inputType == InputType.Unknown:
                            inputType = InputType.Image
                        elif inputType == InputType.TextOnly:
                            inputType = InputType.ImageAndText
                    elif item["type"] == TEXT_CONTENT_TYPE:
                        if inputType == InputType.Unknown:
                            inputType = InputType.TextOnly
                        elif inputType == InputType.Image:
                            inputType = InputType.ImageAndText

        return inputType

    def modify(self, request_obj: any) -> any:
        """Modify the request object."""
        inputType = InputTypeModifier.get_input_type(request_obj=request_obj)
        if inputType != InputType.Unknown:
            request_obj[InputTypeModifier.__INPUT_TYPE_PROPERTY] = inputType
        else:
            lu.get_logger().critical("Could not classify input type")

        return request_obj
