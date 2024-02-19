# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers Config."""

from enum import Enum

from mlflow.types import DataType


MLFLOW_ARTIFACT_DIRECTORY = "mlflow_model_folder"


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Tasks supported for All Frameworks."""

    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"
    MM_MULTI_OBJECT_TRACKING = "video-multi-object-tracking"


class MMDetLiterals:
    """MMDetection constants."""

    CONFIG_PATH = "config_path"
    WEIGHTS_PATH = "weights_path"
    AUGMENTATIONS_PATH = "augmentations_path"
    METAFILE_PATH = "model_metadata"
    CLASSES = "classes"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_IMAGE_KEY = "image_base64"
    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_VIDEO_DATA_TYPE = DataType.string
    INPUT_COLUMN_VIDEO = "video"

    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_FILENAME = "filename"
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"
    OUTPUT_COLUMN_BOXES = "boxes"
    BATCH_SIZE_KEY = "batch_size"
    SCHEMA_SIGNATURE = "signature"
    TRAIN_LABEL_LIST = "train_label_list"
    WRAPPER = "images_model_wrapper"
    THRESHOLD = "threshold"
    TEXT_PROMPT = "text_prompt"
    CUSTOM_ENTTIES = "custom_entities"


class ODLiterals:
    """OD constants."""

    LABEL = "label"
    BOXES = "boxes"
    SCORE = "score"
    BOX = "box"
    TOP_X = "topX"
    TOP_Y = "topY"
    BOTTOM_X = "bottomX"
    BOTTOM_Y = "bottomY"


class ISLiterals:
    """IS constants."""

    POLYGON = "polygon"


class MOTLiterals(ODLiterals):
    """MMTracking dataset constants."""

    DET_BBOXES = "det_bboxes"
    DET_LABELS = "det_labels"
    TRACK_BBOXES = "track_bboxes"
    TRACK_LABELS = "track_labels"
    INSTANCE_ID = "instance_id"

    IMG_INFO = "img_info"
    FRAME_ID = "frame_id"
    VIDEO_URL = "video_url"
