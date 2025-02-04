# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Student Model Info and Requirements."""

import re
from typing import Dict, List

from common.constants import REGISTRY_MODEL_PATTERN, DataGenerationTaskType


class StudentModels:
    """Student model information and requirements."""

    SUPPORTED_STUDENT_MODELS = {
        "Meta-Llama-3.1-8B-Instruct": {
            "supported_registries": ["azureml-meta"],
            "supported_version_pattern": re.compile(r"\d+")
        },
        "Phi-3-mini-4k-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+")
        },
        "Phi-3-mini-128k-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+")
        },
        "Phi-3.5-mini-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+")
        },
        "Phi-3.5-MoE-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+")
        },
        "Phi-3-medium-4k-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+"),
        },
        "Phi-3-medium-128k-instruct": {
            "supported_registries": ["azureml"],
            "supported_version_pattern": re.compile(r"\d+"),
        },
    }

    # Student models that do not recognize system prompts
    NO_SYSTEM_PROMPT_MODELS = [
        "Phi-3-medium-4k-instruct",
        "Phi-3-medium-128k-instruct"
    ]

    @classmethod
    def no_system_prompt_reformat(cls, data: List[Dict[str, list]]) -> List[Dict[str, list]]:
        """Add system prompt to user prompt for student models that do not accept system prompts.

        :param data: The synthetic data generated from the teacher model
        :type data: List[Dict[str, list]]
        :return: Reformated data
        :rtype: List[Dict[str, list]]
        """
        new_data = []
        system_message = ""
        for messages in data:
            system_message = messages["messages"][0]["content"]
            question = messages["messages"][1]["content"]
            reformatted_data = {
                "messages":
                    [
                        {"role": "user", "content": system_message + " " + question},
                        messages["messages"][2]
                    ]
            }
            new_data.append(reformatted_data)
        return new_data

    @classmethod
    def no_system_prompt_reformat_conversation(cls, data: List[Dict[str, list]]) -> List[Dict[str, list]]:
        """Add system prompt to user prompt for student models that do not accept system prompts.

        :param data: The synthetic data generated from the teacher model
        :type data: List[Dict[str, list]]
        :return: Reformated data
        :rtype: List[Dict[str, list]]
        """
        new_data = []
        system_message = ""
        for messages in data:
            system_message = messages["messages"][0]["content"]
            user_prompt = messages["messages"][1]["content"]
            reformatted_data = {
                "messages":
                    [
                        {"role": "user", "content": system_message + " " + user_prompt},
                        messages["messages"][2:]
                    ]
            }
            new_data.append(reformatted_data)
        return new_data

    @classmethod
    def reformat(cls, student_model: str, task_type: str, data: List[Dict[str, list]]) -> List[Dict[str, list]]:
        """Reformats synthetic data based on the student model and task type requirements.

        :param student_model: The student model to finetune
        :type student_model: str
        :param task_type: The data generation task type
        :type task_type: str
        :param data: The synthetic data generated from the teacher model
        :type data: List[Dict[str, list]]
        :return: Reformatted data based on student model and task type
        :rtype: List[Dict[str, list]]
        """
        if student_model in cls.NO_SYSTEM_PROMPT_MODELS:
            if task_type == DataGenerationTaskType.CONVERSATION:
                return cls.no_system_prompt_reformat_conversation(data)
            return cls.no_system_prompt_reformat(data)
        return data

    @classmethod
    def parse_model_asset_id(cls, asset_id: str) -> str:
        """Parse asset id to extract the student model name.

        :param asset_id: The asset id of the student model in the form
            azureml://registries/{registry}/models/{model}/versions/{version}.
        :type asset_id: str
        """
        match = re.search(REGISTRY_MODEL_PATTERN, asset_id)
        model = match.group("model")

        if model not in cls.NO_SYSTEM_PROMPT_MODELS:
            raise Exception("Model is not in supported student model list")
        return model
