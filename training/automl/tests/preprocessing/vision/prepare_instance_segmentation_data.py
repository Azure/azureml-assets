# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image Instance Segmentation preprocessing."""

import os
from azure.ai.ml import MLClient
from tempfile import TemporaryDirectory
from vision.utils import _download_and_register_image_data


def _create_jsonl_files(data_dir, uri_folder_data_path, src_images):
    from jsonl_converter import convert_mask_in_VOC_to_jsonl


    convert_mask_in_VOC_to_jsonl(src_images, uri_folder_data_path)


def prepare_data(mlclient: MLClient):
    """Prepare image OD data.

    :param mlclient: mlclient object to upload and register datasets
    :type mlclient: MLClient
    """

    data_dir = os.path.join(os.getcwd(), "automl/tests/test_configs/assets/image-instance-segmentation-fridge-items")
    instance_segmentation_fridge_items_url = (
        "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjectsMask.zip"
    )

    with TemporaryDirectory() as tempdir:
        local_path, uri_folder_path = _download_and_register_image_data(
            mlclient, instance_segmentation_fridge_items_url, tempdir, "odFridgeObjectsMask")
        _create_jsonl_files(data_dir, uri_folder_path, local_path)
