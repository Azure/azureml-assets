# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preprocess model."""

import azureml.model.mgmt.processors.transformers as transformers
from azureml.model.mgmt.config import ModelFlavor
from azureml.model.mgmt.processors import pyfunc
from azureml.model.mgmt.utils.logging_utils import logger
from pathlib import Path
from typing import Dict


logger = logger.get_logger(__name__)


def run_preprocess(mlflow_flavor: str, model_path: Path, output_dir: Path, temp_dir: Path, **preprocess_args: Dict):
    """Preprocess model.

    :param mlflow_flavor: mlflow flavor for converting model to
    :type mlflow_flavor: str
    :param model_path: input model path
    :type model_path: Path
    :param output_dir: directory where converted mlflow model would be saved to
    :type output_dir: Path
    :param temp_dir: directory for temporary operations
    :type output_dir: Path
    :param preprocess_args: additional preprocess args required by mlflow flavor
    :type preprocess_args: Dict
    """
    logger.info(f"Run preprocess for model with flavor: {mlflow_flavor} at path: {model_path}")
    if mlflow_flavor == ModelFlavor.TRANSFORMERS.value:
        transformers.to_mlflow(model_path, output_dir, temp_dir, preprocess_args)
    elif mlflow_flavor == ModelFlavor.MMLAB_PYFUNC.value:
        pyfunc.to_mlflow(model_path, output_dir, preprocess_args)
    else:
        raise Exception(f"Unsupported model flavor: {mlflow_flavor}.")
    logger.info("Model prepocessing completed !!!")
