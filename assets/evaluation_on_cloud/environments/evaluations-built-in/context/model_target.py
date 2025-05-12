# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module for handling model targets in evaluation on cloud."""
import requests
import logging


USER = "User"
ASSISTANT = "Assistant"


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_key_from_dict(d, key, default=None):
    """Get a particular Key from given dict"""
    for k, v in d.items():
        if k.lower() == key.lower():
            return v
    return default


class ModelTarget:
    """Represents a model target and runs the chat/completion API on a query based on the target."""

    def __init__(self, endpoint, api_key, model_params, system_message, few_shot_examples):
        """Initialize ModelTarget."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_params = model_params
        self.system_message = system_message
        self.few_shot_examples = few_shot_examples

    def generate_response(self, query, context=None, **kwargs):
        """Invoke the model target with the given input query."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key
        }

        messages = [{"role": "system", "content": self.system_message}]
        for example in self.few_shot_examples:
            user_content = get_key_from_dict(example, USER, None)
            if user_content is not None:
                messages.append({"role": "user", "content": user_content})
            assistant_content = get_key_from_dict(example, ASSISTANT, None)
            if assistant_content is not None:
                messages.append({"role": "assistant", "content": assistant_content})

        if context:
            messages.append({"role": "user", "content": f"{context} {query}"})
        else:
            messages.append({"role": "user", "content": query})

        payload = {
            "messages": messages,
            **self.model_params
        }

        response = requests.post(self.endpoint, headers=headers, json=payload)

        if response.status_code == 200:
            try:
                output = response.json()
                response_text = output["choices"][0]["message"]["content"]
                return response_text
            except (KeyError, IndexError):
                raise RuntimeError(f"Error: Unexpected API response format: {KeyError}")
        else:
            raise RuntimeError(f"Error: {response.status_code}, {response.text}")
