# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""PyFunc convert model."""

from azureml.model.mgmt.processors.pyfunc.convertors import PyFuncMLFLowConvertor
from azureml.model.mgmt.processors.pyfunc.factory import get_mlflow_convertor
from azureml.model.mgmt.utils.common_utils import log_execution_time
from pathlib import Path
from typing import Dict


@log_execution_time
def to_mlflow(input_dir: Path, output_dir: Path, temp_dir: Path, translate_params: Dict):
    """Convert pytorch model to PyFunc MLflow flavor."""
    mlflow_convertor: PyFuncMLFLowConvertor = get_mlflow_convertor(
        model_dir=input_dir, output_dir=output_dir, temp_dir=temp_dir, translate_params=translate_params
    )
    mlflow_convertor.save_as_mlflow()
