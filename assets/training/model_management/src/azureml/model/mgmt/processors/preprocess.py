# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preprocess model."""

from azureml.model.mgmt.processors.convertors import MLFLowConvertorInterface
from azureml.model.mgmt.processors.factory import get_mlflow_convertor
from azureml.model.mgmt.utils.logging_utils import get_logger
from pathlib import Path
from typing import Dict


logger = get_logger(__name__)


def run_preprocess(model_framework: str, model_path: Path, output_dir: Path, temp_dir: Path, **preprocess_args: Dict):
    """Preprocess model.

    :param model_framework: Model framework
    :type model_framework: str
    :param model_path: input model path
    :type model_path: Path
    :param output_dir: directory where converted MLflow model would be saved to
    :type output_dir: Path
    :param temp_dir: directory for temporary operations
    :type output_dir: Path
    :param preprocess_args: additional preprocess args required by MLflow flavor
    :type preprocess_args: Dict
    """
    logger.info(f"Run preprocess for model from framework: {model_framework} at path: {model_path}")
    mlflow_convertor: MLFLowConvertorInterface = get_mlflow_convertor(
        model_framework=model_framework, model_dir=model_path, output_dir=output_dir, temp_dir=temp_dir,
        translate_params=preprocess_args
    )
    mlflow_convertor.save_as_mlflow()
    logger.info("Model preprocessing completed.")
