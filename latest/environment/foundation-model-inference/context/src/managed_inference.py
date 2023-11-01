# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides the MIRPayload class that codifies the payload that is received in the scoring script."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from configs import SerializableDataClass
from constants import TaskType
from logging_config import configure_logger

logger = configure_logger(__name__)


@dataclass
class MIRPayload(SerializableDataClass):
    """Json serializable dataclass that represents the input received from the server."""

    query: Union[str, List[str]]
    params: Dict[str, Any]
    task_type: str

    @classmethod
    def from_dict(cls, mir_input_data: Dict):
        """Create an instance of MIRPayload from input data received from the server."""
        query, params, task_type = get_request_data(mir_input_data)
        return MIRPayload(query, params, task_type)

    def convert_query_to_list(self) -> None:
        """
        Convert the query parameter into a list.

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

    Taken from:
    https://github.com/facebookresearch/llama/blob/main/llama/generation.py

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
    SPECIAL_TAGS = ["[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"]
    UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."

    def process_dialog(messages) -> Tuple[str, List[Tuple[str, str]], str]:
        system_prompt = ""
        user_assistant_messages = []  # list of (user, assistant) messages
        user_message = None  # current user message being processed
        last_user_message = None  # user prompt for which response is needed

        unsafe_request = any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in messages])
        if unsafe_request:
            raise Exception(UNSAFE_ERROR)

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "system" and i == 0:
                system_prompt = content
            elif role == "user":
                if i == len(messages) - 1:
                    last_user_message = content
                else:
                    user_message = content
            elif role == "assistant" and user_message is not None:
                user_assistant_messages.append((user_message, content))
                user_message = None

        return system_prompt, user_assistant_messages, last_user_message

    # ref: https://huggingface.co/spaces/huggingface-projects/\
    # llama-2-7b-chat/blob/main/model.py
    def format_chat_conv(message: str, chat_history: List[Tuple[str, str]], system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'] \
            if system_prompt != "" \
            else ['<s>[INST] ']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(
                f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    sys_prompt, user_assistant_msgs, message = process_dialog(dialog)
    chat_conv = format_chat_conv(message, user_assistant_msgs, sys_prompt)
    return chat_conv


def get_request_data(data) -> (Tuple)[Union[str, List[str]], Dict[str, Any], str]:
    """Process and validate inference request.

    return type for chat-completion: str, dict, str
    return type for text-generation: list, dict, str
    """
    try:
        inputs = data.get("input_data", None)
        task_type = data.get("task_type", TaskType.TEXT_GENERATION)
        if task_type == "chat-completion":
            task_type = TaskType.CONVERSATIONAL

        input_data = []  # type: Union[str, List[str]]
        params = {}  # type: Dict[str, Any]

        if not isinstance(inputs, dict):
            raise Exception("Invalid input data")

        input_data = inputs["input_string"]
        params = inputs.get("parameters", {})

        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == TaskType.CONVERSATIONAL:
            logger.info("chat-completion task. Processing input data")
            input_data = get_processed_input_data_for_chat_completion(input_data)

        return input_data, params, task_type
    except Exception as e:
        raise Exception(
            json.dumps(
                {
                    "error": (
                        "Expected input format: \n"
                        '{"input_data": {"input_string": "<query>", '
                        '"parameters": {"k1":"v1", "k2":"v2"}}}.\n '
                        "<query> should be in below format:\n "
                        'For text-generation: ["str1", "str2", ...]\n'
                        'For chat-completion: [{"role":"user", "content": "str1"},'
                        '{"role": "assistant", "content": "str2"} ....]'
                    ),
                    "exception": str(e),
                }
            )
        )
