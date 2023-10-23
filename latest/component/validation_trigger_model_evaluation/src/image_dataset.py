# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML ACFT Image evaluation component - image dataset."""


from __future__ import annotations
import base64
import json
import pandas as pd
import torch
import numpy as np

from PIL import Image
from torch import Tensor
from typing import cast, Dict, Tuple

import constants

from image_constants import SettingLiterals, ImageDataFrameParams, ODISLiterals
from logging_utilities import get_logger

from azureml.automl.core.shared.constants import MLTableLiterals, MLTableDataLabel

from azureml.acft.common_components.image.runtime_common.common import (
    utils,
)
from azureml.acft.common_components.image.runtime_common.common.aml_dataset_base_wrapper import (
    AmlDatasetBaseWrapper,
)
from azureml.acft.common_components.image.runtime_common.classification.io.read.dataset_wrappers import (
    AmlDatasetWrapper,
)
from azureml.acft.common_components.image.runtime_common.object_detection.common import (
    masktools
)
from azureml.acft.common_components.image.runtime_common.object_detection.data.dataset_wrappers import (
    CommonObjectDetectionDatasetWrapper,
    DatasetProcessingType,
)
from azureml.acft.common_components.image.runtime_common.object_detection.data import (
    datasets,
)
from azureml.acft.common_components.image.runtime_common.object_detection.data.datasets import (
    CommonObjectDetectionDataset,
)
from azureml.core import Workspace
from azureml.core.run import Run

logger = get_logger(name=__name__)


def get_workspace() -> Workspace:
    """Get workspace.

    :return: Workspace
    """
    try:
        ws = Run.get_context().experiment.workspace
    except Exception:
        ws = Workspace.from_config()
    return ws


class RuntimeDetectionDatasetAdapter(CommonObjectDetectionDatasetWrapper):
    """Dataset adapter class that makes Runtime dataset classes suitable for finetune components."""

    def __init__(self, dataset: CommonObjectDetectionDataset) -> None:
        """Dataset adapter class that makes Runtime dataset classes suitable for finetune components. \
        It prepares the input parameters and directs the call to corresponding methods in inherited class. \
        It also modifies the output (before returning) to make it more generic and suitable for finetune components.

        :param dataset: Common object detection dataset
        :type dataset: CommonObjectDetectionDataset.
        """
        # Since, we don't want to apply any augmentation from runtime dataset, setting following values.
        # We will apply augmentation/ pre-processing from finetune components.
        dataset.apply_automl_train_augmentations = False
        dataset._transform = None

        super().__init__(dataset, DatasetProcessingType.IMAGES)

    def __getitem__(self, index: int) -> Tuple[Tensor, dict, dict]:
        """Convert output of dataset get item to make it generalized and usable in components.

        :param index: Index of object
        :type index: int
        :return: Image tensor in de-normalized form [0-255], training labels and image info
        :rtype: Tuple[Tensor, dict, dict]
        """
        image, training_labels, image_info = super().__getitem__(index)

        if image is None:
            return None, {}, {}

        # CommonObjectDetectionDatasetWrapper returns the normalized image. This adapter returns
        # the image in generic de-normalized format to the frameworks (MMD need image in denormalized format).

        with torch.no_grad():
            image = torch.mul(image, 255)
        image = image.to(torch.uint8)

        training_labels[ODISLiterals.CLASSES] = training_labels[ODISLiterals.LABELS].numpy()
        training_labels[ODISLiterals.LABELS] = np.array([self._dataset.index_to_label(x)
                                                         for x in training_labels[ODISLiterals.LABELS]])

        training_labels[ODISLiterals.BOXES] = training_labels[ODISLiterals.BOXES].numpy()
        # rle masks need for computing metrics
        if ODISLiterals.MASKS in training_labels:
            training_labels[ODISLiterals.MASKS] = [masktools.encode_mask_as_rle(mask)
                                                   for mask in training_labels[ODISLiterals.MASKS]]
        return image, training_labels, image_info


def _combine_mltables(training_mltable: str, validation_mltable: str = None) -> str:
    """Combine mltables to make single mltable to pass in get_tabular_dataset.

    :param training_mltable: The training mltable path
    :param validation_mltable: The validation mltable path
    :return: mltable in serialized json format
    """
    mltable = {MLTableDataLabel.TrainData.value: {MLTableLiterals.MLTABLE_RESOLVEDURI: training_mltable}}
    if validation_mltable is not None:
        mltable[MLTableDataLabel.ValidData.value] = {MLTableLiterals.MLTABLE_RESOLVEDURI: validation_mltable}
    return json.dumps(mltable)


def is_valid_image(image_path):
    """Check if image is valid.

    :param image_path: The image path
    """
    try:
        with Image.open(image_path) as img:
            if len(img.getbands()) != 3:
                return False
    except Exception:
        return False
    return True


def read_image(image_path):
    """Read image from path.

    :param image_path: The image path
    """
    with open(image_path, "rb") as f:
        return f.read()


def get_classification_dataset(
    testing_mltable: str,
    settings: Dict = {},
    multi_label: bool = False,
) -> AmlDatasetWrapper:
    """
    Return training and validation dataset for classification task from mltable.

    :param testing_mltable: The training mltable path
    :param settings: Settings dictionary
    :param multi_label: True if multi label classification, False otherwise
    :return: Data Frame with test image paths and labels
    """
    mltable = _combine_mltables(testing_mltable)

    dataset_wrapper: AmlDatasetBaseWrapper = cast(AmlDatasetBaseWrapper, AmlDatasetWrapper)

    ws = get_workspace()

    test_tabular_ds, valid_tabular_ds = utils.get_tabular_dataset(settings=settings, mltable_json=mltable)

    utils.download_or_mount_image_files(
        settings=settings,
        train_ds=test_tabular_ds,
        validation_ds=valid_tabular_ds,
        dataset_class=dataset_wrapper,
        workspace=ws,
    )

    label_column_name = settings.get(SettingLiterals.LABEL_COLUMN_NAME, None)
    test_dataset_wrapper = AmlDatasetWrapper(
        test_tabular_ds,
        multilabel=multi_label,
        label_column_name=label_column_name,
    )

    logger.info(
        f"# test images: {len(test_dataset_wrapper)}, \
        # labels: {test_dataset_wrapper.num_classes}"
    )

    df = pd.DataFrame(columns=[ImageDataFrameParams.IMAGE_COLUMN_NAME, ImageDataFrameParams.LABEL_COLUMN_NAME])
    for index in range(len(test_dataset_wrapper)):
        image_path = test_dataset_wrapper.get_image_full_path(index)
        if is_valid_image(image_path):
            df = df.append({
                ImageDataFrameParams.IMAGE_COLUMN_NAME: base64.encodebytes(read_image(image_path)).decode("utf-8"),
                ImageDataFrameParams.LABEL_COLUMN_NAME: test_dataset_wrapper.label_at_index(index)
            }, ignore_index=True)

    return df


def get_object_detection_dataset(
    test_mltable: str,
    settings: Dict = {},
    masks_required: bool = False,
) -> Tuple[RuntimeDetectionDatasetAdapter, RuntimeDetectionDatasetAdapter]:
    """Return training and validation dataset for object detection and instance segmentation task from mltable.

    :param training_mltable: The training mltable path
    :type training_mltable: str
    :param object_detection_dataset: The dataset adapter class name to be used for creating dataset objects.
    :type object_detection_dataset: RuntimeDetectionDatasetAdapter
    :param settings: Settings dictionary
    :type settings: Dict
    :param validation_mltable: The validation mltable path
    :type validation_mltable: str
    :param masks_required: mask required or not for segmentation. Optional, default False
    :type masks_required: bool
    :return: Training dataset, validation dataset
    :rtype: Tuple[RuntimeDetectionDatasetAdapter, RuntimeDetectionDatasetAdapter]
    """
    mltable = _combine_mltables(test_mltable, test_mltable)

    dataset_wrapper: AmlDatasetBaseWrapper = cast(
        AmlDatasetBaseWrapper, datasets.AmlDatasetObjectDetection
    )
    test_tabular_ds, _ = utils.get_tabular_dataset(
        settings=settings, mltable_json=mltable
    )

    ws = get_workspace()

    utils.download_or_mount_image_files(
        settings=settings,
        train_ds=test_tabular_ds,
        validation_ds=None,
        dataset_class=dataset_wrapper,
        workspace=ws,
    )
    logger.info("# downloaded test images")

    use_bg_label = settings.get(SettingLiterals.USE_BG_LABEL, False)
    ignore_data_errors = settings.get(SettingLiterals.IGNORE_DATA_ERRORS, True)

    test_dataset = datasets.AmlDatasetObjectDetection(dataset=test_tabular_ds, is_train=False,
                                                      ignore_data_errors=ignore_data_errors,
                                                      settings=settings, use_bg_label=use_bg_label,
                                                      masks_required=masks_required,)
    logger.info(
        f"# test images: {len(test_dataset)}, # labels: {test_dataset.num_classes}"
    )
    test_dataset_wrapper = RuntimeDetectionDatasetAdapter(test_dataset)
    df = pd.DataFrame(columns=[ImageDataFrameParams.IMAGE_COLUMN_NAME,
                               ImageDataFrameParams.LABEL_COLUMN_NAME,
                               ImageDataFrameParams.IMAGE_META_INFO])

    counter = 0
    for index in range(len(test_dataset_wrapper)):
        _, label, image_meta_info = test_dataset_wrapper[index]
        image_path = test_dataset_wrapper._dataset._dataset_elements[index].image_url

        if is_valid_image(image_path):
            counter += 1
            df = df.append({
                ImageDataFrameParams.IMAGE_COLUMN_NAME: base64.encodebytes(read_image(image_path)).decode("utf-8"),
                ImageDataFrameParams.LABEL_COLUMN_NAME: label,
                ImageDataFrameParams.IMAGE_META_INFO: image_meta_info
            }, ignore_index=True)

    logger.info(f"Total number of valid images: {counter}")
    return df


def get_image_dataset(task_type, test_mltable, settings={}):
    """
    Return test dataset for image tasks from mltable.

    :param testing_mltable: The training mltable path
    :param settings: Settings dictionary
    :param multi_label: True if multi label classification, False otherwise
    :return: Data Frame with test image paths and labels
    """
    if task_type in [constants.TASK.IMAGE_CLASSIFICATION, constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL]:
        multi_label = True if task_type == constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL else False
        return get_classification_dataset(
            testing_mltable=test_mltable,
            settings=settings,
            multi_label=multi_label,
        )
    elif task_type in [constants.TASK.IMAGE_OBJECT_DETECTION, constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
        masks_required = True if task_type == constants.TASK.IMAGE_INSTANCE_SEGMENTATION else False
        return get_object_detection_dataset(
            test_mltable=test_mltable,
            settings=settings,
            masks_required=masks_required,
        )
    else:
        raise ValueError(f"Task type {task_type} not supported")
