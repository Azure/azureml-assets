# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from .vesta_chat_completion_image_modifier import VestaChatCompletionImageModifier
from .vesta_image_encoder import ImageEncoder


class VestaChatCompletionEncodedImageScrubber(RequestModifier):
    def modify(self, request_obj: any) -> any:
        if "Column1" in request_obj:
            request_obj = json.loads(request_obj["Column1"])

        lu.get_logger().debug("Image Scrubber: Input data type = {}, Input data = {}".format(type(request_obj), request_obj))
        if VestaChatCompletionImageModifier.is_vesta_chat_completion_payload(request_obj=request_obj):
            for message in request_obj["messages"]:
                for content in message["content"]:
                    if isinstance(content, dict):
                        if "image" in content:
                            content["image"] = self._scrub_image_encoding(content["image"])
                        if "image_hr" in content:
                            content["image_hr"] = self._scrub_image_encoding(content["image_hr"])
            return request_obj
        else:
            lu.get_logger().error("Input data does not match Vesta chat completion schema")
            raise Exception("Input data does not match Vesta chat completion schema")

    def _scrub_image_encoding(self, image_data: str):
        if not image_data.startswith(ImageEncoder.IMAGE_URL) and not image_data.startswith(ImageEncoder.IMAGE_FILE):
            return "<Encoded image data has been scrubbed>"

        return image_data
