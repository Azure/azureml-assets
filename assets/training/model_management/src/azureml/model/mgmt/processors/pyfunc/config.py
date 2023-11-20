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

    # CLIP task
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
    EMBEDDINGS = "embeddings"

    # BLIP tasks
    IMAGE_TO_TEXT = "image-to-text"
    VISUAL_QUESTION_ANSWERING = "visual-question-answering"

    # Text to Image
    TEXT_TO_IMAGE = "text-to-image"
    TEXT_TO_IMAGE_INPAINTING = "text-to-image-inpainting"
    IMAGE_TEXT_TO_IMAGE = "image-text-to-image"

    # LLaVA task
    IMAGE_TEXT_TO_TEXT = "image-text-to-text"

    # mask generation
    MASK_GENERATION = "mask-generation"


class SupportedTextToImageModelFamily(_CustomEnum):
    """Supported text to image models."""

    STABLE_DIFFUSION = "stable-diffusion"
