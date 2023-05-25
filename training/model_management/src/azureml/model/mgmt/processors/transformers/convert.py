# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers convert model."""

from azureml.model.mgmt.processors.transformers.convertors import HFMLFLowConvertor
from azureml.model.mgmt.processors.transformers.factory import get_mlflow_convertor
from azureml.model.mgmt.utils.common_utils import log_execution_time
from pathlib import Path
from typing import Dict


@log_execution_time
def to_mlflow(input_dir: Path, output_dir: Path, translate_params: Dict):
    """Convert Hugging face pytorch model to Mlflow."""
    mlflow_convertor: HFMLFLowConvertor = get_mlflow_convertor(
        model_dir=input_dir, output_dir=output_dir, translate_params=translate_params
    )
    mlflow_convertor.save_as_mlflow()
