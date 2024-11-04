# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML ACFT Image evaluation component - image dataset."""


from __future__ import annotations
import base64
import json
import pandas as pd
import torch
import numpy as np

from mltable import load
from PIL import Image
from torch import Tensor
from typing import cast, Dict, List, Tuple

import constants

from image_constants import GenerationLiterals, ODISLiterals, SettingLiterals
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
    input_column_names: List[str],
    label_column_name: str,
    settings: Dict = {},
    multi_label: bool = False,
) -> pd.DataFrame:
    """
    Return training and validation dataset for classification task from mltable.

    :param test_mltable: The path to the prediction input mltable
    :param input_column_names: The column names of the model inputs
    :param label_column_name: The column name of the label
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

    test_dataset_wrapper = AmlDatasetWrapper(
        test_tabular_ds,
        multilabel=multi_label,
        label_column_name=label_column_name,
    )

    logger.info(
        f"# test images: {len(test_dataset_wrapper)}, \
        # labels: {test_dataset_wrapper.num_classes}"
    )

    # Initialize the rows of the output dataframe to the empty list.
    frame_rows = []

    for index in range(len(test_dataset_wrapper)):
        image_path = test_dataset_wrapper.get_image_full_path(index)
        if is_valid_image(image_path):
            # sending image_paths instead of base64 encoded string as oss flavor doesnt take bytes as input.
            frame_rows.append({
                input_column_names[0]: image_path,
                label_column_name: test_dataset_wrapper.label_at_index(index)
            })

    # Make the output dataframe.
    df = pd.DataFrame(data=frame_rows, columns=input_column_names + [label_column_name])

    return df


def get_object_detection_dataset(
    test_mltable: str,
    input_column_names: List[str],
    label_column_name: str,
    settings: Dict = {},
    masks_required: bool = False,
) -> pd.DataFrame:
    """Return training and validation dataset for object detection and instance segmentation task from mltable.

    :param test_mltable: The path to the prediction input mltable
    :param input_column_names: The column names of the model inputs
    :param label_column_name: The column name of the label
    :param settings: Settings dictionary
    :param masks_required: mask required or not for segmentation. Optional, default False
    :return: Data Frame with test image paths and labels
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

    # Initialize the rows of the output dataframe to the empty list.
    frame_rows = []

    counter = 0
    for index in range(len(test_dataset_wrapper)):
        _, label, image_meta_info = test_dataset_wrapper[index]
        image_path = test_dataset_wrapper._dataset._dataset_elements[index].image_url

        if is_valid_image(image_path):
            counter += 1
            frame_rows.append({
                input_column_names[0]: base64.encodebytes(read_image(image_path)).decode("utf-8"),
                input_column_names[1]: image_meta_info,
                input_column_names[2]: ". ".join([str(c) for c in test_dataset.classes]),
                label_column_name: label,
            })

    # Make the output dataframe.
    df = pd.DataFrame(data=frame_rows, columns=input_column_names + [label_column_name])

    logger.info(f"Total number of valid images: {counter}")
    return df


def get_generation_dataset(
    mltable_path: str,
    input_column_names: List[str],
    label_column_name: str,
    settings: Dict = {},
):
    """
    Make input dataset for image generation from mltable.

    :param test_mltable: The path to the prediction input mltable
    :param input_column_names: The column names of the model inputs
    :param label_column_name: The column name of the label
    :param settings: Settings dictionary
    :return: Data Frame with test image paths and labels
    """
    # Workaround for MLTable not being able to convert image url from stream back to string.
    full_mltable_file_name = mltable_path + "/" + SettingLiterals.MLTABLE_FILE_NAME
    with open(full_mltable_file_name, "rt") as f:
        mltable_str = f.read()
    mltable_str = mltable_str.replace(SettingLiterals.MLTABLE_STREAM_STR, "")
    with open(full_mltable_file_name, "wt") as f:
        f.write(mltable_str)

    # Load MLTable and convert to Pandas dataframe.
    mltable = load(mltable_path)
    mltable_dataframe = mltable.to_pandas_dataframe()

    # Initialize the rows of the output dataframe to the empty list.
    frame_rows = []

    # Go through all (image_url, captions) pairs and make a (prompt, image_url) from each pair. The model will generate
    # a synthetic image from the prompt and the set of synthetic images will be compared with the set of original ones.
    for image_url, captions in zip(
        mltable_dataframe[SettingLiterals.IMAGE_URL], mltable_dataframe[SettingLiterals.LABEL]
    ):
        # Go through all captions (split according to special separator).
        for caption in captions.split(GenerationLiterals.CAPTION_SEPARATOR):
            frame_rows.append(
                {
                    # The model input is a text prompt.
                    input_column_names[0]: caption,
                    # The original image is passed through via the label column.
                    label_column_name: image_url,
                }
            )

    # Make the output dataframe.
    df = pd.DataFrame(data=frame_rows, columns=input_column_names + [label_column_name])

    return df


def get_image_dataset(task_type, test_mltable, input_column_names, label_column_name, settings={}):
    """Return test dataset for image tasks from mltable.

    Important details: for vision datasets, the MLTable must have columns "image_url" and "label". For some tasks, the
    output Pandas dataframe may have other column names to respect the model input expectations.

    :param task_type: The type of the prediction task
    :param test_mltable: The path to the prediction input mltable
    :param input_column_names: The column names of the model inputs
    :param label_column_name: The column name of the label
    :param settings: Settings dictionary
    :return: Data Frame with image paths and labels
    """
    if task_type in [constants.TASK.IMAGE_CLASSIFICATION, constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL]:
        multi_label = True if task_type == constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL else False
        return get_classification_dataset(
            testing_mltable=test_mltable,
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            settings=settings,
            multi_label=multi_label,
        )
    elif task_type in [constants.TASK.IMAGE_OBJECT_DETECTION, constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
        masks_required = True if task_type == constants.TASK.IMAGE_INSTANCE_SEGMENTATION else False
        return get_object_detection_dataset(
            test_mltable=test_mltable,
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            settings=settings,
            masks_required=masks_required,
        )
    elif task_type == constants.TASK.IMAGE_GENERATION:
        return get_generation_dataset(
            mltable_path=test_mltable,
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            settings=settings,
        )
    else:
        raise ValueError(f"Task type {task_type} not supported")
