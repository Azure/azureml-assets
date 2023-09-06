from enum import Enum
from mlflow.types import DataType

class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """
    Task types supported by stable diffusion
    """
    TEXT_TO_IMAGE = "text-to-image"


class ColumnNames:
    """
    Column names in pandas dataframe used to receive request and send response.
    """
    TEXT_PROMPT = "prompt"
    GENERATED_IMAGE = "image"
    NSFW_FLAG = "nsfw_content_detected"


class DatatypeLiterals:
    """
    Literals related to data type.
    """
    IMAGE_FORMAT = "JPEG"
    STR_ENCODING = "utf-8"


class MLflowLiterals:
    """
    MLflow export related literals
    """
    MODEL_DIR = "model_dir"
    MODEL_NAME = "model_name"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    IMAGE_DATA_TYPE = DataType.binary
    STRING_DATA_TYPE = DataType.string
