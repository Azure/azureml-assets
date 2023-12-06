# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from .vesta_image_encoder import ImageEncoder, VestaImageModificationException


class VestaChatCompletionImageModifier(RequestModifier):
    @staticmethod
    def is_vesta_chat_completion_payload(request_obj: any):
        return ("messages" in request_obj) and all("content" in message for message in request_obj["messages"])

    def __init__(self, image_encoder: "ImageEncoder" = None) -> None:
        self.__image_encoder: ImageEncoder = image_encoder

    def modify(self, request_obj: any) -> any:
        if "Column1" in request_obj:
            request_obj = json.loads(request_obj["Column1"])

        lu.get_logger().debug("Image Modifier: Input data type = {}, Input data = {}".format(type(request_obj), request_obj))
        if VestaChatCompletionImageModifier.is_vesta_chat_completion_payload(request_obj=request_obj):
            for message in request_obj["messages"]:
                for content in message["content"]:
                    if isinstance(content, dict):
                        if "image" in content:
                            content["image"] = self._modify_image(image_data=content["image"])
                        if "image_hr" in content:
                            content["image_hr"] = self._modify_image(image_data=content["image_hr"])
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
