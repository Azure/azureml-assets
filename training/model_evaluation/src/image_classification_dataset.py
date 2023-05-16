# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML ACFT Image evaluation component - image classification dataset."""

from __future__ import annotations
import base64
import json
import pandas as pd

from PIL import Image
from typing import cast, Dict

from azureml.automl.core.shared.constants import MLTableLiterals, MLTableDataLabel

from azureml.acft.common_components import get_logger_app
from azureml.acft.common_components.image.runtime_common.common import (
    utils,
)
from azureml.acft.common_components.image.runtime_common.common.aml_dataset_base_wrapper import (
    AmlDatasetBaseWrapper,
)
from azureml.acft.common_components.image.runtime_common.classification.io.read.dataset_wrappers import (
    AmlDatasetWrapper,
)

from azureml.core import Workspace
from azureml.core.run import Run

logger = get_logger_app(__name__)


class SettingLiterals:
    """Setting literals for classification dataset."""

    LABEL_COLUMN_NAME = "label_column_name"


class ImageDataFrameParams:
    """DataFrame parameters for classification dataset."""

    IMAGE_COLUMN_NAME = "image"
    LABEL_COLUMN_NAME = "label"


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

    try:
        ws = Run.get_context().experiment.workspace
    except Exception:
        ws = Workspace.from_config()

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

    def is_valid_image(image_path):
        try:
            img = Image.open(image_path)
            if len(img.getbands()) != 3:
                return False
        except Exception:
            return False
        return True

    def read_image(image_path):
        with open(image_path, "rb") as f:
            return f.read()

    df = pd.DataFrame(columns=[ImageDataFrameParams.IMAGE_COLUMN_NAME, ImageDataFrameParams.LABEL_COLUMN_NAME])
    for index in range(len(test_dataset_wrapper)):
        image_path = test_dataset_wrapper.get_image_full_path(index)
        if is_valid_image(image_path):
            df = df.append({ImageDataFrameParams.IMAGE_COLUMN_NAME: \
                                base64.encodebytes(read_image(image_path)).decode("utf-8"),
                            ImageDataFrameParams.LABEL_COLUMN_NAME: test_dataset_wrapper.label_at_index(index)
                            }, ignore_index=True)

    return df
