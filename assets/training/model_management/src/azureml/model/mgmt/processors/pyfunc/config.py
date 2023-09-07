# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyFunc Config."""

from enum import Enum

from azureml.model.mgmt.config import _CustomEnum


class SupportedVisionTasks(_CustomEnum):
    """Supported Vision tasks."""

    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"


class SupportedTasks(_CustomEnum):
    """Supported tasks for conversion to PyFunc MLflow."""

    # Vision tasks
    MM_OBJECT_DETECTION = "image-object-detection"
    MM_INSTANCE_SEGMENTATION = "image-instance-segmentation"

    # CLIP task
    ZERO_SHOT_IMAGE_CLASSIFICATION = "zero-shot-image-classification"
