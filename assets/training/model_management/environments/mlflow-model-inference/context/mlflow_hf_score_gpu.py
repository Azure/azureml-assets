"""Mlflow hugging face gpu."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import logging
import torch
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.models import Model
from mlflow.protos.databricks_pb2 import (
    RESOURCE_DOES_NOT_EXIST,
)
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path
)
from azureml.contrib.services.aml_response import AMLResponse
from azureml.evaluate.mlflow.hftransformers import _load_pyfunc as azureml_lp

FLAVOR_NAME = "python_function"
CODE = "code"
DATA = "data"


_logger = logging.getLogger(__name__)


class SupportedTasks:
    """Supported Tasks."""

    def __init__(self):
        """Init."""
        self.supported_tasks = {
            "LARGE_LANGUAGE_TASKS": [
                "question-answering",
                "text-classification",
                "fill-mask",
                "summarization",
                "text-generation",
                "token-classification",
                "translation",
            ],
            "AUDIO_TASKS": [
                "automatic-speech-recognition",
            ],
            "VISION_TASKS": [
                "image-classification",
                "image-classification-multilabel",
            ]
        }

    def large_language(self):
        """Large language."""
        return self.supported_tasks['LARGE_LANGUAGE_TASKS']

    def audio(self):
        """Audio."""
        return self.supported_tasks["AUDIO_TASKS"]

    def vision(self):
        """Vision."""
        return self.supported_tasks["VISION_TASKS"]

    def all(self):
        """All."""
        return self.audio() + self.vision() + self.large_language()

    # Audio and Vision models have loader modules for predict
    def media(self):
        """Audio and Vision models have loader modules for predict."""
        return self.audio() + self.vision()


# Function that converts pandas dataframe input to json
def convert_pandas_to_dict(input_data):
    """For Function that converts pandas dataframe input to json."""
    return input_data.to_dict() if ("inputs" in input_data.columns and
                                    len(input_data.columns) == 1) else {"inputs": input_data.to_dict()}


# Function that appends device parameter
def append_device_parameter(inputs):
    """For Function that appends device parameter."""
    device_parameter = {"dev_args": {"device": 0}}
    if "parameters" in inputs:
        inputs["parameters"].update(device_parameter)
    else:
        inputs.update({"parameters": device_parameter})

    return inputs


# Function that validates translation types
def validate_translation_type(translation_type):
    """For Function that validates translation types."""
    if translation_type == "translation":
        return True
    # Translation tasks should have one of the following formats: "translation_xx_to_yy"
    # or "translation_xx_yy_to_zz_aa"
    translation_pattern = re.compile(
        "|".join(
            [
                "^translation_[A-Za-z]{2}_to_[A-Za-z]{2}$",
                "^translation_[A-Za-z]{2}_[A-Za-z]{2}_to_[A-Za-z]{2}_[A-Za-z]{2}$",
            ]
        )
    )
    return bool(translation_pattern.match(translation_type))


def init():
    """Init."""
    global task_name, predict, signature, supported_tasks

    if torch.cuda.is_available():
        _logger.info("---Using GPU---")
    else:
        raise Exception("---GPU is not available. Please check the CUDA configuration---")

    model_path = str(os.getenv("AZUREML_MODEL_DIR"))

    mlflow_model_folders = list()
    for root, dirs, files in os.walk(model_path):
        for name in files:
            if name.lower() == "mlmodel":
                mlflow_model_folders.append(root)

    if len(mlflow_model_folders) == 0:
        raise Exception("---- No MLmodel files found in AZUREML_MODEL_DIR ----")
    elif len(mlflow_model_folders) > 1:
        print("---- More than one MLmodel files found in AZUREML_MODEL_DIR. Terminating. ----")

    model_path = mlflow_model_folders[0]

    local_path = _download_artifact_from_uri(artifact_uri=model_path)
    model_meta = Model.load(os.path.join(local_path, MLMODEL_FILE_NAME))
    conf = model_meta.flavors.get(FLAVOR_NAME)
    if conf is None:
        raise MlflowException(
            f'Model does not have the "{FLAVOR_NAME}" flavor',
            RESOURCE_DOES_NOT_EXIST,
        )

    _add_code_from_conf_to_system_path(local_path, conf, code_key=CODE)
    data_path = os.path.join(local_path, conf[DATA]) if (DATA in conf) else local_path
    kwargs = {"model_hf_load_kwargs": {"device_map": "eval_na"}}
    model_impl = azureml_lp(data_path, **kwargs)
    task_name = model_impl.task_type
    supported_tasks = SupportedTasks()
    if not (task_name in supported_tasks.all() or validate_translation_type(task_name)):
        return AMLResponse(
            f"Invalid task_name, task should be one of the following: {supported_tasks.all()}",
            400,
        )
    predict = model_impl.predict
    signature = model_meta.signature


# Function that handles real-time inference requests
def online_inference(input_data):
    """For Function that handles real-time inference requests."""
    if isinstance(input_data, pd.DataFrame):
        input_data = convert_pandas_to_dict(input_data)

    if isinstance(input_data, dict):
        parameters = input_data.get("parameters", {})
        translation_type = parameters.get("task_type", "")

        if "translation" not in task_name and "translation" in translation_type:
            return AMLResponse("Cannot pass task_type parameter for given model.", 400)
        if "translation" in task_name and "translation" in translation_type:
            if not validate_translation_type(translation_type):
                return AMLResponse(
                    """Invalid translation_type, should be in form translation_ab_to_yz or
                translation_xx_yy_to_zz_aa""",
                    400,
                )
    if isinstance(input_data, pd.DataFrame):
        input_data = convert_pandas_to_dict(input_data)

    input_data = append_device_parameter(input_data)
    try:
        return _get_jsonable_obj(predict(input_data), pandas_orient="records")
    except Exception as e:
        return AMLResponse(str(e), 400)


def run(input_data):
    """Run for input data."""
    _logger.info("Inference request received")

    # Process String input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    if (
        # Accepted input formats are:
        # dictioary with only "inputs" key
        (isinstance(input_data, dict) and "inputs" in input_data and len(input_data.keys()) == 1)
        # dictionary with only "inputs" and "parameters" key
        or (isinstance(input_data, dict) and "inputs" in input_data and "parameters" in input_data
            and len(input_data.keys()) == 2)
        # dataframe with inputs key only
        or (isinstance(input_data, pd.DataFrame) and "inputs" in input_data.columns
            and len(input_data.columns) == 1)
        # dataframe with model signature keys
        or (isinstance(input_data, pd.DataFrame) and
            set(i['name'] for i in signature.inputs.to_dict()) == set(input_data.columns.to_list()))
    ):
        result = online_inference(input_data)
    else:
        return AMLResponse(
            "Invalid input. Use dict in form"
            + """ '{"inputs": {"input_signature":["data"]},"parameters": {<parameters>}}' """
            + """or pandas dataframe in form {"inputs": {"inputs_signature": ["data"]}}""", 400
        )

    _logger.info("Inferencing successful")
    return result
