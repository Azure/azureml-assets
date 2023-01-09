# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image Object Detection preprocessing."""

import json
import logging
import os
import xml.etree.ElementTree as ET
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
        "image_details": {"format": None, "width": None, "height": None},
        "label": [],
    }

    # Path to the annotations
    annotations_folder = os.path.join(src_images, "annotations")

    # Read each annotation and convert it to jsonl line
    with open(train_annotations_file, "w") as train_f:
        with open(validation_annotations_file, "w") as validation_f:
            for i, filename in enumerate(os.listdir(annotations_folder)):
                if filename.endswith(".xml"):
                    logger.info("Parsing " + os.path.join(src_images, filename))

                    root = ET.parse(os.path.join(annotations_folder, filename)).getroot()

                    width = int(root.find("size/width").text)
                    height = int(root.find("size/height").text)

                    labels = []
                    for object in root.findall("object"):
                        name = object.find("name").text
                        xmin = object.find("bndbox/xmin").text
                        ymin = object.find("bndbox/ymin").text
                        xmax = object.find("bndbox/xmax").text
                        ymax = object.find("bndbox/ymax").text
                        isCrowd = int(object.find("difficult").text)
                        labels.append(
                            {
                                "label": name,
                                "topX": float(xmin) / width,
                                "topY": float(ymin) / height,
                                "bottomX": float(xmax) / width,
                                "bottomY": float(ymax) / height,
                                "isCrowd": isCrowd,
                            }
                        )
                    # build the jsonl file
                    image_filename = root.find("filename").text
                    _, file_extension = os.path.splitext(image_filename)
                    json_line = dict(json_line_sample)
                    json_line["image_url"] = json_line["image_url"] + "images/" + image_filename
                    json_line["image_details"]["format"] = file_extension[1:]
                    json_line["image_details"]["width"] = width
                    json_line["image_details"]["height"] = height
                    json_line["label"] = labels

                    if i % train_validation_ratio == 0:
                        # validation annotation
                        validation_f.write(json.dumps(json_line) + "\n")
                    else:
                        # train annotation
                        train_f.write(json.dumps(json_line) + "\n")
                else:
                    logger.info("Skipping unknown file: {}".format(filename))
    logger.info("Created jsonl files")


def prepare_data(mlclient: MLClient):
    """Prepare Image Object Detection data.

    :param mlclient: mlclient object to upload and register datasets
    :type mlclient: MLClient
    """
    data_dir = os.path.join(os.getcwd(), "automl/tests/test_configs/assets/image-object-detection-fridge-items")
    object_detection_fridge_items_url = (
        "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip"
    )

    with TemporaryDirectory() as tempdir:
        local_path, uri_folder_path = _download_and_register_image_data(
            mlclient, object_detection_fridge_items_url, tempdir, "odFridgeObjects"
        )
        _create_jsonl_files(data_dir, uri_folder_path, local_path)
