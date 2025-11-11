# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Managed inference payload module.

This module provides the MIRPayload class that codifies the payload format
received in the scoring script for managed inference requests.
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple, Union, Type, TypeVar

from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.conversation import TextMessage, MultimodalMessage
from foundation.model.serve.constants import EnvironmentVariables, TaskType

logger = configure_logger(__name__)


@dataclass
class SerializableDataClass:
    """A base data class that can be serialized to and from a dictionary."""

    def to_dict(self) -> Dict:
        """Convert the data class to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        """Create a data class instance from a dictionary.

        Args:
            d (Dict): The dictionary containing data class field values.

        Returns:
            An instance of the data class.
        """
        return cls(**d)


@dataclass
class MIRPayload(SerializableDataClass):
    """Managed inference request payload.

    This class represents the JSON serializable dataclass for inference input,
    containing the query, parameters, and task type.
    """

    query: Union[List[TextMessage], List[MultimodalMessage],
                 str, List[str], List[Tuple[str, str]]]
    params: Dict[str, Any]
    task_type: str

    @classmethod
    def from_dict(cls, mir_input_data: Dict):
        """Create an instance of MIRPayload from input data received from the server.

        Args:
            mir_input_data (Dict): The input data dictionary.

        Returns:
            MIRPayload: An instance of MIRPayload.
        """
        query, params, task_type = get_request_data(mir_input_data)
        return MIRPayload(query, params, task_type)

    def convert_query_to_list(self) -> None:
        """Convert the query parameter into a list.

        The inference engine expects a list of prompts. In the case of chat completion,
        a single string is produced and needs to be put inside a list.
        """
        if not isinstance(self.query, list):
            self.query = [self.query]

    def update_params(self, new_params: Dict) -> None:
        """Update current parameters to the new parameters the MIRPayload should have.

        Args:
            new_params (Dict): The new parameters to set.
        """
        self.params = new_params


def get_request_data(
    data
) -> (Tuple)[Union[str, List[str]], Dict[str, Any], str, bool]:
    """Process and validate inference request data.

    Args:
        data: The input data dictionary.

    Returns:
        tuple: A tuple containing (input_data, params, task_type).
            - For chat-completion: (str, dict, str, bool)
            - For text-generation: (list, dict, str, bool)

    Raises:
        Exception: If the input data format is invalid.
    """
    try:
        task_type = os.getenv(
            EnvironmentVariables.TASK_TYPE, TaskType.CHAT_COMPLETION)
        inputs = data.get("input_data", None)

        if task_type != TaskType.TEXT_GENERATION:
            if not isinstance(inputs, dict):
                raise Exception("Invalid input data")

        input_data = []  # type: Union[str, List[str]]
        params = {}  # type: Dict[str, Any]

        # Input format is being updated
        # Original text-gen input: {"input_data": {"input_string": ["<query>"], "parameters": {"k1":"v1", "k2":"v2"}}}
        # New text-gen input: {"input_data": ["<query>"], "params": {"k1":"v1", "k2":"v2"}}
        # For other tasks, new input format is not provided yet, so keeping original format for now.
        is_text_gen_without_input = (
            task_type == TaskType.TEXT_GENERATION and "input_string" not in inputs
        )
        input_data = inputs if is_text_gen_without_input else inputs["input_string"]
        params = data.get("params", {}) if is_text_gen_without_input else inputs.get(
            "parameters", {})

        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        return input_data, params, task_type
    except Exception as e:
        task_type = data.get(EnvironmentVariables.TASK_TYPE,
                             TaskType.CHAT_COMPLETION)
        if task_type == TaskType.CHAT_COMPLETION:
            correct_input_format = (
                '{"input_data": {"input_string": [{"role":"user", "content": "str1"}, '
                '{"role": "assistant", "content": "str2"} ....], "parameters": {"k1":"v1", "k2":"v2"}}}'
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
