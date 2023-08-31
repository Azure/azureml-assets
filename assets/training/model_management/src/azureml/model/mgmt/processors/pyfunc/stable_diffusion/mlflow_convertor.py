# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import os
from pathlib import Path
from typing import Dict

from .mlflow_wrapper import StableDiffusionMLflowWrapper
from azureml.model.mgmt.config import ComponentConstants
from azureml.model.mgmt.utils.common_utils import log_execution_time


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

    mlflow.pyfunc.save_model(
        path=output_dir,
        python_model=mlflow_model_wrapper,
        pip_requirements=pip_requirements,
        code_path=code_path,
        metadata={"model_name": model_id},
    )
