"""Mlflow hugging face for cpu."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import logging
import pandas as pd
import azureml.evaluate.mlflow as azureml_mlflow
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azureml.contrib.services.aml_response import AMLResponse


_logger = logging.getLogger(__name__)


class SupportedTasks:
    """For Support Taks."""

    def __init__(self):
        """For init ."""
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
            ],
        }

    def large_language(self):
        """Large language."""
        return self.supported_tasks["LARGE_LANGUAGE_TASKS"]

    def audio(self):
        """For Audio."""
        return self.supported_tasks["AUDIO_TASKS"]

    def vision(self):
        """For Vision."""
        return self.supported_tasks["VISION_TASKS"]

    def all(self):
        """For all."""
        return self.audio() + self.vision() + self.large_language()

    def media(self):
        """For media."""
        return self.audio() + self.vision()


# Function that sanitizes pandas dataframe input
def sanitize_pandas_input(input_data):
    """For Function that sanitizes pandas dataframe input."""
    if isinstance(input_data, pd.DataFrame) and "inputs" in input_data.columns and len(input_data.columns) == 1:
        return input_data.to_dict()
    else:
        pass


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
    """For init."""
    global task_name, model, supported_tasks

    model_path = str(os.getenv("AZUREML_MODEL_DIR"))

    # Walking through the AZUREML_MODEL_DIR folder to find folder containing MLmodel file.
    # Terminates if number of MLmodel files != 1
    mlflow_model_folders = list()
    for root, dirs, files in os.walk(model_path):
        for name in files:
            if name.lower() == "mlmodel":
                mlflow_model_folders.append(root)

    if len(mlflow_model_folders) == 0:
        raise Exception("---- No MLmodel files found in AZUREML_MODEL_DIR ----")
    elif len(mlflow_model_folders) > 1:
        raise Exception("---- More than one MLmodel files found in AZUREML_MODEL_DIR. Terminating. ----")

    model_path = mlflow_model_folders[0]

    model = azureml_mlflow.pyfunc.load_model(model_path)
    task_name = model._model_impl.task_type

    # Initialize SupportedTasks
    supported_tasks = SupportedTasks()

    if task_name not in supported_tasks.all() or not validate_translation_type(task_name):
        return AMLResponse(
            f"Invalid task_name, task should be one of the following: {supported_tasks.all()}",
            400,
        )


# Function that handles real-time inference requests
def online_inference(input_data):
    """For Function that handles real-time inference requests."""
    if isinstance(input_data, pd.DataFrame):
        model_input_signature = model.metadata.signature.inputs.to_dict()
        if len(model_input_signature) > 1 or task_name in supported_tasks.media():
            input_data = sanitize_pandas_input(input_data)

    if isinstance(input_data, dict):
        parameters = input_data.get("parameters", {})
        translation_type = parameters.get("task_type", "")

        if "translation" not in task_name and "translation" in translation_type:
            return AMLResponse("Cannot pass task_type parameter for given model.", 400)
        if "translation" in task_name and "translation" in translation_type:
            if not validate_translation_type(translation_type):
                return AMLResponse(
                    """Invalid translation_type, should be in form translation_ab_to_yz or
                translation_ab_cd_to_wx_yz""",
                    400,
                )
    try:
        return _get_jsonable_obj(model._model_impl.predict(input_data), pandas_orient="records")
    except Exception as e:
        return AMLResponse(str(e), 400)


def run(input_data):
    """To run for input data."""
    _logger.info("Inference request received")

    # Process string input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)

    if (
        # Allowed input formats are:
        # dictioary with only "inputs" key
        (isinstance(input_data, dict) and "inputs" in input_data and len(input_data.keys()) == 1)
        # dictionary with "inputs" and "parameters" keys only
        or (
            isinstance(input_data, dict)
            and "inputs" in input_data
            and "parameters" in input_data
            and len(input_data.keys()) == 2
        )
        # dataframe with "inputs" column
        or (isinstance(input_data, pd.DataFrame) and "inputs" in input_data.columns and len(input_data.columns) == 1)
        # dataframe with model input signature as columns
        or (
            isinstance(input_data, pd.DataFrame)
            and set(i["name"] for i in model.metadata.signature.inputs.to_dict()) == set(input_data.columns.to_list())
        )
    ):
        result = online_inference(input_data)
    else:
        return AMLResponse(
            "Invalid input. Use dict in form"
            + """ '{"inputs": {"input_signature":["data"]},"parameters": {<parameters>}}' """
            + """or pandas dataframe in form {"inputs": {"input_signature1": ["data1"], "input_signature2":["data2"]
              .... }}""",
            400,
        )

    _logger.info("Inferencing successful")
    return result
