# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Image constants for evaluation."""


class SettingLiterals:
    """Setting literals for classification dataset."""

    IMAGE_URL = "image_url"
    LABEL = "label"
    MASKS_REQUIRED = "masks_required"
    USE_BG_LABEL = "use_bg_label"
    IGNORE_DATA_ERRORS = "ignore_data_errors"
    BOX_SCORE_THRESHOLD = "box_score_threshold"
    IOU_THRESHOLD = "iou_threshold"
    MLTABLE_FILE_NAME = "MLTable"
    MLTABLE_STREAM_STR = "- convert_column_types:\n  - column_type: stream_info\n    columns: image_url\n"


class ImageDataConstants:
    """Data constants."""

    DEFAULT_BOX_SCORE_THRESHOLD = 0.3
    DEFAULT_IOU_THRESHOLD = 0.5


class ImageDataFrameParams:
    """DataFrame parameters for image dataset."""

    IMAGE_COLUMN_NAME = "image"
    LABEL_COLUMN_NAME = "label"
    IMAGE_META_INFO = "image_meta_info"
    PREDICTIONS = "predictions"
    TEXT_PROMPT = "text_prompt"
    GENERATION_PROMPT = "prompt"


class ODISLiterals:
    """Object detection and instance segmentation literals."""

    BOXES = "boxes"
    BOX = "box"
    CLASSES = "classes"
    CLASS = "class"
    SCORES = "scores"
    SCORE = "score"
    LABELS = "labels"
    LABEL = "label"
    TOP_X = "topX"
    TOP_Y = "topY"
    BOTTOM_X = "bottomX"
    BOTTOM_Y = "bottomY"
    NUM_CLASSES = "num_classes"
    MASKS = "masks"
    POLYGON = "polygon"
    HEIGHT = "height"
    WIDTH = "width"
    LABEL_INDEX = "label_index"


class GenerationLiterals:
    """Image generation literals."""

    CAPTION_SEPARATOR = "||"
