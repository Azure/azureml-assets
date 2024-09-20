# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vesta chat completion image modifier."""

import json

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from .vesta_image_encoder import ImageEncoder, VestaImageModificationException


class VestaChatCompletionImageModifier(RequestModifier):
    """Vesta chat completion image modifier."""

    @staticmethod
    def is_vesta_chat_completion_payload(request_obj: any):
        """Check whether the payload is in Vesta chat completion format."""
        return ("messages" in request_obj) and all("content" in message for message in request_obj["messages"])

    def __init__(self, image_encoder: "ImageEncoder" = None) -> None:
        """Initialize VestaChatCompletionImageModifier."""
        self.__image_encoder: ImageEncoder = image_encoder

    def modify(self, request_obj: any) -> any:
        """Modify the request object."""
        if "Column1" in request_obj:
            request_obj = json.loads(request_obj["Column1"])

        if VestaChatCompletionImageModifier.is_vesta_chat_completion_payload(request_obj=request_obj):
            for message in request_obj["messages"]:
                for content in message["content"]:
                    if isinstance(content, dict):
                        if "image" in content:
                            content["image"] = self._modify_image(image_data=content["image"])
                        if "image_hr" in content:
                            content["image_hr"] = self._modify_image(image_data=content["image_hr"])
                        if "image_url" in content and "url" in content["image_url"]:
                            content["image_url"]["url"] = self._modify_image(image_data=content["image_url"]["url"])
            return request_obj
        else:
            lu.get_logger().error("Input data does not match Vesta chat completion schema")
            raise Exception("Input data does not match Vesta chat completion schema")

    def _modify_image(self, image_data: str):
        try:
            return self.__image_encoder.encode_b64(image_data)
        except Exception as e:
            lu.get_logger().error(f"ImageEncoder raised an exception: {e}")
            raise VestaImageModificationException() from e
