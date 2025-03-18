# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyFunc Config."""

from azureml.model.mgmt.config import _CustomEnum


class MMLabDetectionTasks(_CustomEnum):
    """Supported tasks from MMLab model framework for detection."""

    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"


class MMLabTrackingTasks(_CustomEnum):
    """Supported tasks from MMLab model framework for video."""

    MM_MULTI_OBJECT_TRACKING = "video-multi-object-tracking"


class SupportedTasks(_CustomEnum):
    """Supported tasks for conversion to PyFunc MLflow."""

    # MMLab detection tasks
    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"

    # MmTrack video tasks
    MM_MULTI_OBJECT_TRACKING = "video-multi-object-tracking"

    # CLIP tasks
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"

    # Embedding tasks
    EMBEDDINGS = "embeddings"

    # BLIP tasks
    IMAGE_TO_TEXT = "image-to-text"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"

    # Text to Image
    TEXT_TO_IMAGE = "text-to-image"
    TEXT_TO_IMAGE_INPAINTING = "text-to-image-inpainting"
    IMAGE_TO_IMAGE = "image-to-image"

    # LLaVA task
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"

    # mask generation
    MASK_GENERATION = "mask-generation"

    # AutoML Tasks
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_CLASSIFICATION_MULTILABEL = "image-classification-multilabel"
    IMAGE_OBJECT_DETECTION = "image-object-detection"
    IMAGE_INSTANCE_SEGMENTATION = "image-instance-segmentation"

    # Virchow
    IMAGE_FEATURE_EXTRACTION = "image-feature-extraction"

    # Hibou-B family
    FEATURE_EXTRACTION = "feature-extraction"


class ModelFamilyPrefixes(_CustomEnum):
    """Prefixes for some of the models converted to PyFunc MLflow."""

    # CLIP model family.
    CLIP = "openai/clip-vit"

    # DinoV2 model family.
    DINOV2 = "facebook/dinov2"

    # Virchow model family.
    VIRCHOW = "paige-ai/Virchow"

    # Hibou-B family
    HIBOU_B = "histai/hibou-b"
