# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image Classification Multilabel preprocessing."""

import json
import logging
import os
from azure.ai.ml import MLClient
from tempfile import TemporaryDirectory
from vision.utils import _download_and_register_image_data


logger = logging.Logger(__name__)


def _create_jsonl_files(data_dir, uri_folder_data_path, src_images):
    # We'll copy each JSONL file within its related MLTable folder
    training_mltable_path = os.path.join(data_dir, "./training-mltable-folder/")
    validation_mltable_path = os.path.join(data_dir, "./validation-mltable-folder/")

    train_validation_ratio = 5

    # Path to the training and validation files
    train_annotations_file = os.path.join(training_mltable_path, "train_annotations.jsonl")
    validation_annotations_file = os.path.join(validation_mltable_path, "validation_annotations.jsonl")

    # Baseline of json line dictionary
    json_line_sample = {
        "image_url": uri_folder_data_path,
        "label": [],
    }

    # Path to the labels file.
    labelFile = os.path.join(src_images, "labels.csv")

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            with open(labelFile, "r") as labels:
                for i, line in enumerate(labels):
                    # Skipping the title line and any empty lines.
                    if i == 0 or len(line.strip()) == 0:
                        continue
                    line_split = line.strip().split(",")
                    if len(line_split) != 2:
                        logger.info("Skipping the invalid line: {}".format(line))
                        continue
                    json_line = dict(json_line_sample)
                    json_line["image_url"] += f"images/{line_split[0]}"
                    json_line["label"] = line_split[1].strip().split(" ")

                    if i % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")


def prepare_data(mlclient: MLClient):
    """Prepare Image Classification Multilabel data.

    :param mlclient: mlclient object to upload and register datasets
    :type mlclient: MLClient
    """
    data_dir = os.path.join(
        os.getcwd(), "automl/tests/test_configs/assets/image-classification-multilabel-fridge-items"
    )
    classification_multilabel_fridge_items_url = (
        "https://cvbp-secondary.z19.web.core.windows.net/datasets/image_classification/multilabelFridgeObjects.zip"
    )

    with TemporaryDirectory() as tempdir:
        local_path, uri_folder_path = _download_and_register_image_data(
            mlclient, classification_multilabel_fridge_items_url, tempdir, "multilabelFridgeObjects"
        )
        _create_jsonl_files(data_dir, uri_folder_path, local_path)
