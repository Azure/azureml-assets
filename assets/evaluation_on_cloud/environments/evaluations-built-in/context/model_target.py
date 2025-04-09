# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module for handling model targets in evaluation on cloud."""
import requests
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
            if 'User' in example:
                messages.append({"role": "user", "content": example['User']})
            if 'Assistant' in example:
                messages.append({"role": "assistant", "content": example['Assistant']})

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
                return "Error: Unexpected API response format"
        else:
            raise RuntimeError(f"Error: {response.status_code}, {response.text}")
