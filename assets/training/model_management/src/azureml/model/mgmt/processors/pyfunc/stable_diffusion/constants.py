from enum import Enum

class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """
    Task types supported by stable diffusion
    """
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_INPAINTING = "image-inpainting"


class COLUMN_NAMES:
    """
    Column names in pandas dataframe used to receive request and send response.
    """
    TEXT_PROMPTS = "text_prompts"
    GENERATED_IMAGES = "generated_images"

class DATATYPE_LITERALS:
    """
    Literals related to data type.
    """
    IMAGE_FORMAT = "JPEG"
    STR_ENCODING = "utf-8"
