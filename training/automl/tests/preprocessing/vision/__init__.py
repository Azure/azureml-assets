# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision Test preprocessing init file."""

from .prepare_classification_data import prepare_data as prepare_classification_data
from .prepare_classification_multilabel_data import prepare_data as prepare_classification_multilabel_data
from .prepare_instance_segmentation_data import prepare_data as prepare_instance_segmentation_data
from .prepare_object_detection_data import prepare_data as prepare_object_detection_data
