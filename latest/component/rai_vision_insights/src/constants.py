# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from enum import Enum


class TaskType:
    IMAGE_CLASSIFICATION = "image_classification"
    MULTILABEL_IMAGE_CLASSIFICATION = "multilabel_image_classification"
    OBJECT_DETECTION = "object_detection"


class ImageColumns(str, Enum):
    """Provide constants related to the input image dataframe columns.

    Can be 'image_url', 'image' or 'label'.
    """

    IMAGE_URL = 'image_url'
    IMAGE = 'image'
    LABEL = 'label'


class DataPaths:
    BLOB_STORAGE_URL = "https://publictestdatasets.blob.core.windows.net/"
    FRIDGE_OD_BLOB_URL = BLOB_STORAGE_URL +\
        "computervision/odFridgeObjects/images/"
    FRIDGE_OD_DOWNLOAD_URL = (
        "https://publictestdatasets.blob.web.core.windows.net/" +
        "computervision/odFridgeObjects.zip"
    )
    FRIDGE_OD_DOWNLOAD_URL_DATASETS = (
        "https://publictestdatasets.blob.core.windows.net/" +
        "computervision/odFridgeObjects_transformed.zip"
    )
    FRIDGE_OD_DATA_FILE = "./odFridgeObjects.zip"
    FRIDGE_OD_DATA_FILE_DATASETS = "./odFridgeObjects_transformed.zip"
    FRIDGE_OD_DIRECTORY = "data"
    FRIDGE_OD_SRC_IMAGES = "./data/odFridgeObjects/"
    FRIDGE_OD_SRC_IMAGES_DATASETS = "./data/odFridgeObjects_transformed/"
