# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers convert model."""

import json
import os
import sys
from pathlib import Path
from typing import Dict

import mlflow
from azureml.model.mgmt.utils.common_utils import log_execution_time
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from .vision.config import MLFlowSchemaLiterals, MMDetLiterals, Tasks


def _prepare_artifacts_dict(input_dir: Path) -> Dict:
    """Prepare artifacts dict for mlflow model.

    :param input_dir: input directory
    :type input_dir: Path
    :return: artifacts dict
    :rtype: Dict
    """
    metadata_path = os.path.join(input_dir, "model_selector_args.json")
    with open(metadata_path) as f:
        metadata = json.load(f)

    artifacts_dict = {
        MMDetLiterals.CONFIG_PATH: os.path.join(input_dir, metadata.get("pytorch_model_path")),
        MMDetLiterals.WEIGHTS_PATH: os.path.join(input_dir, metadata.get("model_weights_path_or_url")),
    }
    return artifacts_dict


def _get_mlflow_signature(task_type: str) -> ModelSignature:
    """Return mlflow model signature with input and output schema given the input task type.

    :param task_type: Task type used in training
    :type task_type: str
    :return: mlflow model signature.
    :rtype: mlflow.models.signature.ModelSignature
    """
    input_schema = Schema(
        [ColSpec(MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE_DATA_TYPE, MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE)]
    )

    if task_type == Tasks.MM_OBJECT_DETECTION.value:
        output_schema = Schema(
            [
                ColSpec(MLFlowSchemaLiterals.OUTPUT_COLUMN_DATA_TYPE, MLFlowSchemaLiterals.OUTPUT_COLUMN_BOXES),
            ]
        )
    else:
        raise NotImplementedError(f"Task type: {task_type} is not supported yet.")
    return ModelSignature(inputs=input_schema, outputs=output_schema)


@log_execution_time
def to_mlflow(input_dir: Path, output_dir: Path, translate_params: Dict) -> None:
    """Convert pytorch model to Mlflow.

    :param input_dir: model input directory
    :type input_dir: Path
    :param output_dir: output directory
    :type output_dir: Path
    :param translate_params: mlflow translation params
    :type translate_params: Dict
    """
    model_name = translate_params.get("model_id")
    task = translate_params["task"]

    sys.path.append(os.path.join(os.path.dirname(__file__), "vision"))
    from detection_predict import ImagesDetectionMLFlowModelWrapper

    mlflow_model_wrapper = ImagesDetectionMLFlowModelWrapper(task_type=task)
    artifacts_dict = _prepare_artifacts_dict(input_dir)
    pip_requirements = os.path.join(os.path.dirname(__file__), "vision", "requirements.txt")
    code_path = [
        os.path.join(os.path.dirname(__file__), "vision", "detection_predict.py"),
        os.path.join(os.path.dirname(__file__), "vision", "config.py"),
    ]
    signatures = translate_params.get("signature") or _get_mlflow_signature(task)

    mlflow.pyfunc.save_model(
        path=output_dir,
        python_model=mlflow_model_wrapper,
        artifacts=artifacts_dict,
        pip_requirements=pip_requirements,
        signature=signatures,
        code_path=code_path,
        metadata={"model_name": model_name},
    )

    print("Model saved!!!")
