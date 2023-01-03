# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image Instance Segmentation preprocessing."""

import os
from azure.ai.ml import MLClient
from tempfile import TemporaryDirectory
from vision.utils import _download_and_register_image_data


_DATA_DIR = os.path.join(os.getcwd(), "automl/tests/test_configs/assets/image-instance-segmentation-fridge-items")
_INSTANCE_SEGMENTATION_FRIDGE_ITEMS_URL = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip"


def _create_jsonl_files(uri_folder_data_path, src_images):
    from jsonl_converter import convert_mask_in_VOC_to_jsonl


    convert_mask_in_VOC_to_jsonl(src_images, uri_folder_data_path)


def prepare_data(mlclient: MLClient):
    """Prepare image OD data.

    :param mlclient: mlclient object to upload and register datasets
    :type mlclient: MLClient
    """

    with TemporaryDirectory() as tempdir:
        local_path, uri_folder_path = _download_and_register_image_data(mlclient, _INSTANCE_SEGMENTATION_FRIDGE_ITEMS_URL, tempdir, "odFridgeObjectsMask")
        _create_jsonl_files(uri_folder_path, local_path)
