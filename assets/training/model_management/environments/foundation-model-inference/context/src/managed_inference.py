# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This module provides the MIRPayload class that codifies the payload that is received in the scoring script."""
import json
import os
from dataclasses import dataclass
import pandas as pd
from typing import Any, Dict, List, Tuple, Union

from configs import SerializableDataClass
from constants import TaskType
from logging_config import configure_logger
from transformers import AutoTokenizer

logger = configure_logger(__name__)

DEFAULT_TOKENIZER_PATH = "mlflow_model_folder/components/tokenizer"
DEFAULT_MODEL_PATH = "mlflow_model_folder/data/model"


@dataclass
class MIRPayload(SerializableDataClass):
    """Json serializable dataclass that represents the input received from the server."""

    query: Union[str, List[str]]
    params: Dict[str, Any]
    task_type: str
    is_preview_format: bool

    @classmethod
    def from_dict(cls, mir_input_data: Dict):
        """Create an instance of MIRPayload from input data received from the server."""
        query, params, task_type, is_preview_format = get_request_data(mir_input_data)
        return MIRPayload(query, params, task_type, is_preview_format)

    def convert_query_to_list(self) -> None:
        """Convert the query parameter into a list.

        FMScore.run expects a list of prompts. In the case of chat completion, a single string
        is produced and needs to be put inside of a list.
        """
        if not isinstance(self.query, list):
            self.query = [self.query]

    def update_params(self, new_params: Dict) -> None:
        """Update current parameters to the new parameters the MIRPayload should have."""
        self.params = new_params


def get_processed_input_data_for_chat_completion(dialog: List[str]) -> str:
    r"""Process chat completion input request.

    example input:
    [
        {
            "role": "user",
            "content": "What is the tallest building in the world?"
        },
        {
            "role": "assistant",
            "content": "As of 2021, the Burj Khalifa in Dubai"
        },
        {
            "role": "user",
            "content": "and in Africa?"
        },
    ]
    example output:
    "[INST]What is the tallest building in the world?[/INST]
    As of 2021, the Burj Khalifa in Dubai\n
    [INST]and in Africa?[/INST]"
    """
    # get path to model folder
    tokenizer_path = str(os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), DEFAULT_TOKENIZER_PATH))
    if not os.path.exists(tokenizer_path):
        tokenizer_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR", ""),
            DEFAULT_MODEL_PATH,
        )
    # use tokenizer defined in tokenizer_config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_default_system_prompt=False)
    # apply template to format chat conversation
    chat_conv = tokenizer.apply_chat_template(dialog, tokenize=False)
    return chat_conv


def process_input_data_for_text_to_image(inputs: Dict[str, any]) -> Tuple[List[str], Dict[str, Any]]:
    """Process text to image task input request to make it suitable for model.

    :param inputs: input data
    :type inputs: Dict[str, any]
    :raises Exception: if input data is not in expected format
    :return: Processed input data for model and parameters
    :rtype: Tuple[List[str], Dict[str, Any]]
    """
    params = inputs.pop("parameters", {})
    if "columns" in inputs and "data" in inputs:
        input_df = pd.DataFrame(**inputs)
        input_data = input_df["prompt"].to_list()
    return input_data, params


def get_request_data(data) -> (Tuple)[Union[str, List[str]], Dict[str, Any], str, bool]:
    """Process and validate inference request.

    return type for chat-completion: str, dict, str, bool
    return type for text-generation: list, dict, str, bool
    """
    try:
        is_preview_format = True
        inputs = data.get("input_data", None)
        task_type = data.get("task_type", TaskType.TEXT_GENERATION)
        # TODO: Update this check once all tasks are updated to use new input format
        if task_type != TaskType.TEXT_GENERATION:
            if not isinstance(inputs, dict):
                raise Exception("Invalid input data")

        if task_type == "chat-completion":
            task_type = TaskType.CONVERSATIONAL
        elif task_type == TaskType.TEXT_TO_IMAGE:
            input_data, params = process_input_data_for_text_to_image(inputs)
            return input_data, params, task_type, is_preview_format

        input_data = []  # type: Union[str, List[str]]
        params = {}  # type: Dict[str, Any]

        # Input format is being updated
        # Original text-gen input: {"input_data": {"input_string": ["<query>"], "parameters": {"k1":"v1", "k2":"v2"}}}
        # New text-gen input: {"input_data": ["<query>"], "params": {"k1":"v1", "k2":"v2"}}
        # For other tasks, new input format is not provided yet, so keeping original format for now.
        if task_type == TaskType.TEXT_GENERATION and "input_string" not in inputs:
            is_preview_format = False
            input_data = inputs
            params = data.get("params", {})
        else:
            input_data = inputs["input_string"]
            params = inputs.get("parameters", {})

        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == TaskType.CONVERSATIONAL:
            logger.info("chat-completion task. Processing input data")
            input_data = get_processed_input_data_for_chat_completion(input_data)

        return input_data, params, task_type, is_preview_format
    except Exception as e:
        task_type = data.get("task_type", TaskType.TEXT_GENERATION)
        if task_type == "chat-completion":
            correct_input_format = (
                '{"input_data": {"input_string": [{"role":"user", "content": "str1"}, '
                '{"role": "assistant", "content": "str2"} ....], "parameters": {"k1":"v1", "k2":"v2"}}}'
            )
        elif task_type == TaskType.TEXT_TO_IMAGE:
            correct_input_format = (
                '{"input_data": {"columns": ["prompt"], "data": ["prompt sample 1", "prompt sample 2"], '
                '"parameters": {"k1":"v1", "k2":"v2"}}}'
            )
        else:
            correct_input_format = (
                '{"input_data": ["str1", "str2", ...], '
                '"params": {"k1":"v1", "k2":"v2"}}'
            )

        raise Exception(
            json.dumps(
                {
                    "error": (
                        "Expected input format: \n" + correct_input_format
                    ),
                    "exception": str(e),
                },
            ),
        )
