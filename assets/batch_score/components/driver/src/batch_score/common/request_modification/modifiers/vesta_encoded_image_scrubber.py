from ...telemetry import logging_utils as lu
from .request_modifier import RequestModifier
from .vesta_image_modifier import ImageEncoder, VestaImageModifier


class VestaEncodedImageScrubber(RequestModifier):
    def modify(self, request_obj: any) -> any:
        if VestaImageModifier.is_vesta_payload(request_obj=request_obj):
            for transcript_dict in request_obj[VestaImageModifier.vesta_payload_type(request_obj=request_obj)]:
                if transcript_dict["type"] == "image" or transcript_dict["type"] == "image_hr":
                    transcript_dict["data"] = self._scrub_image_encoding(transcript_dict["data"])
            return request_obj
        else:
            lu.get_logger().error("Input data does not match Vesta schema")
            raise Exception("Input data does not match Vesta schema")

    def _scrub_image_encoding(self, image_data: str):
        if not image_data.startswith(ImageEncoder.IMAGE_URL) and not image_data.startswith(ImageEncoder.IMAGE_FILE):
            return "<Encoded image data has been scrubbed>"

        return image_data
