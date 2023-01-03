# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image Classification preprocessing."""

import json
import os
import xml.etree.ElementTree as ET
from azure.ai.ml import MLClient
from tempfile import TemporaryDirectory
from vision.utils import _download_and_register_image_data


_DATA_DIR = os.path.join(os.getcwd(), "automl/tests/test_configs/assets/image-classification-fridge-items")
_CLASSIFICATION_FRIDGE_ITEMS_URL = "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/fridgeObjects.zip"


def _create_jsonl_files(uri_folder_data_path, src_images):    
    # We'll copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join(_DATA_DIR, "./training-mltable-folder/")
    validation_mltable_path = os.path.join(_DATA_DIR, "./validation-mltable-folder/")

    train_validation_ratio = 5

    # Path to the training and validation files
    train_annotations_file = os.path.join(training_mltable_path, "train_annotations.jsonl")
    validation_annotations_file = os.path.join(
        validation_mltable_path, "validation_annotations.jsonl"
    )

    # Baseline of json line dictionary
    json_line_sample = {
        "image_url": uri_folder_data_path,
        "label": "",
    }

    index = 0
    # Scan each sub directary and generate a jsonl line per image, distributed on train and valid JSONL files
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            for className in os.listdir(src_images):
                subDir = src_images + className
                if not os.path.isdir(subDir):
                    continue
                # Scan each sub directary
                print("Parsing " + subDir)
                for image in os.listdir(subDir):
                    json_line = dict(json_line_sample)
                    json_line["image_url"] += f"{className}/{image}"
                    json_line["label"] = className

                    if index % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
                    index += 1


def prepare_data(mlclient: MLClient):
    """Prepare image OD data.

    :param mlclient: mlclient object to upload and register datasets
    :type mlclient: MLClient
    """

    with TemporaryDirectory() as tempdir:
        local_path, uri_folder_path = _download_and_register_image_data(mlclient, _CLASSIFICATION_FRIDGE_ITEMS_URL, tempdir, "multilabelFridgeObjects")
        _create_jsonl_files(uri_folder_path, local_path)
