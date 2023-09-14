# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Convert StableDiffusion model from HuggingFace to MLflow format."""

import os
import shutil
import sys
import yaml

import mlflow

from pathlib import Path
from typing import Dict

from mlflow.models import ModelSignature
from mlflow.types import DataType
from mlflow.types.schema import ColSpec
from mlflow.types.schema import Schema

from constants import MLflowLiterals, MLflowSchemaLiterals, Tasks
from azureml.model.mgmt.utils.common_utils import log_execution_time


def _get_mlflow_signature(task_type: str) -> ModelSignature:
    """Return MLflow model signature with input and output schema given the input task type.
    :param task_type: Task type used in training
    :type task_type: str
    :return: MLflow model signature.
    :rtype: mlflow.models.signature.ModelSignature
    """
    if task_type in [Tasks.IMAGE_TEXT_TO_TEXT.value]:
        input_schema = Schema(
            [
                ColSpec(DataType.string, MLflowSchemaLiterals.INPUT_COLUMN_IMAGE),
                ColSpec(DataType.string, MLflowSchemaLiterals.INPUT_COLUMN_PROMPT),
                ColSpec(DataType.string, MLflowSchemaLiterals.INPUT_COLUMN_DIRECT_QUESTION),
            ]
        )
    else:
        raise NotImplementedError(f"Task type: {task_type} is not supported.")

    output_schema = Schema(
        [
            ColSpec(DataType.string, MLflowSchemaLiterals.OUTPUT_COLUMN_RESPONSE)
        ]
    )

    return ModelSignature(inputs=input_schema, outputs=output_schema)


@log_execution_time
def to_mlflow(input_dir: Path, output_dir: Path, translate_params: Dict) -> None:
    """Convert pytorch model to MLflow.
    :param input_dir: model input directory
    :type input_dir: Path
    :param output_dir: output directory
    :type output_dir: Path
    :param translate_params: MLflow translation params
    :type translate_params: Dict
    :return: None
    """
    task = translate_params["task"]

    current_directory_name = os.path.dirname(__file__)

    # This to get Wrapper class independently and not as part of parent package.
    sys.path.append(os.path.dirname(__file__))
    from llava_mlflow_wrapper import LLaVAMLflowWrapper

    mlflow_model_wrapper = LLaVAMLflowWrapper(task_type=task)

    conda_yaml_path = os.path.join(current_directory_name, "conda.yaml")
    conda_env = {}
    with open(conda_yaml_path) as f:
        conda_env = yaml.safe_load(f)

    code_path = [
        os.path.join(current_directory_name, "llava_mlflow_wrapper.py"),
        os.path.join(current_directory_name, "constants.py"),
        os.path.join(current_directory_name, "utils.py"),
    ]

    model_dir = os.path.join(os.path.dirname(input_dir), "model_dir")
    shutil.copytree(input_dir, model_dir, dirs_exist_ok=True)
    mlflow.pyfunc.save_model(
        path=output_dir,
        python_model=mlflow_model_wrapper,
        artifacts={MLflowLiterals.MODEL_DIR: model_dir},
        signature=_get_mlflow_signature(task),
        conda_env=conda_env,
        code_path=code_path,
        metadata={},
    )
