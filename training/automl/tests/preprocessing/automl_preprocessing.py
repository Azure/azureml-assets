# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AutoML Preprocessing."""

import logging
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from vision import (
    prepare_classification_data,
    prepare_classification_multilabel_data,
    prepare_instance_segmentation_data,
    prepare_object_detection_data,
)


logger = logging.Logger(__name__)

if __name__ == "__main__":
    # preprocessing job list
    preprocessing_jobs = [
        prepare_classification_data.prepare_data,
        prepare_classification_multilabel_data.prepare_data,
        prepare_instance_segmentation_data.prepare_data,
        prepare_object_detection_data.prepare_data,
    ]

    mlclient = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv("subscription_id"),
        resource_group_name=os.getenv("resource_group"),
        workspace_name=os.getenv("workspace"),
    )

    logger.info("Running automl data preprocessing ...")
    for job in preprocessing_jobs:
        job(mlclient=mlclient)
    logger.info("Completed AutoML preprocessing ...")
