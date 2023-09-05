# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import os
import shutil
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec
from mlflow.types.schema import DataType, Schema
from pathlib import Path
from typing import Dict

from .constants import ColumnNames, MLflowLiterals, MLflowSchemaLiterals, Tasks
from .mlflow_wrapper import StableDiffusionMLflowWrapper
from azureml.model.mgmt.config import ComponentConstants
from azureml.model.mgmt.utils.common_utils import log_execution_time



def _get_mlflow_signature(task_type: str) -> ModelSignature:
    """Return MLflow model signature with input and output schema given the input task type.

    :param task_type: Task type used in training
    :type task_type: str
    :return: MLflow model signature.
    :rtype: mlflow.models.signature.ModelSignature
    """

    if task_type in [Tasks.TEXT_TO_IMAGE.value]:
        input_schema = Schema(
            [ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.TEXT_PROMPT)]
        )
    else:
        raise NotImplementedError(f"Task type: {task_type} is not supported yet.")

    output_schema = Schema(
        [ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.TEXT_PROMPT),
         ColSpec(MLflowSchemaLiterals.IMAGE_DATA_TYPE, ColumnNames.GENERATED_IMAGE),
         ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.NSFW_FLAG),
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
    model_id = translate_params.get(ComponentConstants.MODEL_ID)
    task = translate_params[ComponentConstants.TASK]

    mlflow_model_wrapper = StableDiffusionMLflowWrapper(task_type=task, model_id=model_id)
    pip_requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    code_path = [
        os.path.join(os.path.dirname(__file__), "mlflow_wrapper.py"),
        os.path.join(os.path.dirname(__file__), "constants.py"),
    ]

    model_dir = os.path.join(os.path.dirname(input_dir), "model_dir")
    shutil.copytree(input_dir, model_dir, dirs_exist_ok=True)
    mlflow.pyfunc.save_model(
        path=output_dir,
        python_model=mlflow_model_wrapper,
        artifacts={MLflowLiterals.MODEL_DIR: model_dir},
        signature=_get_mlflow_signature(task),
        pip_requirements=pip_requirements,
        code_path=code_path,
        metadata={MLflowLiterals.MODEL_NAME: model_id},
    )
