# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from .vesta_image_encoder import ImageEncoder, VestaImageModificationException


class VestaImageModifier(RequestModifier):
    @staticmethod
    def vesta_payload_type(request_obj: any) -> str:
        payload_type = None

        if "transcript" in request_obj:
            payload_type = "transcript"
        elif "prompt" in request_obj:  # Also supports "prompt" as a key
            payload_type = "prompt"

        if "transcript" in request_obj and "prompt" in request_obj:
            return None  # Either "transcript" or "prompt" should be used, not both

        return payload_type

    @staticmethod
    def is_vesta_payload(request_obj: any):
        payload_type = VestaImageModifier.vesta_payload_type(request_obj=request_obj)

        return (payload_type is not None and
                (payload_type in request_obj) and
                all("type" in transcript for transcript in request_obj[payload_type]))

    def __init__(self, image_encoder: "ImageEncoder" = None) -> None:
        self.__image_encoder: ImageEncoder = image_encoder

    def modify(self, request_obj: any) -> any:
        if VestaImageModifier.is_vesta_payload(request_obj=request_obj):
            for transcript_dict in request_obj[VestaImageModifier.vesta_payload_type(request_obj=request_obj)]:
                if transcript_dict["type"] == "image" or transcript_dict["type"] == "image_hr":
                    try:
                        transcript_dict["data"] = self.__image_encoder.encode_b64(transcript_dict["data"])
                    except Exception as e:
                        lu.get_logger().error(f"ImageEncoder raised an exception: {e}")
                        raise VestaImageModificationException() from e
            return request_obj
        else:
            lu.get_logger().error("Input data does not match Vesta schema")
            raise Exception("Input data does not match Vesta schema")
