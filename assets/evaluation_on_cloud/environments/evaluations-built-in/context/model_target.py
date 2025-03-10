import re
import os
import requests
import logging
from azure.core.exceptions import HttpResponseError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTarget:
    def __init__(self, endpoint, api_key, model_params, system_message):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model_params = model_params
        self.system_message = system_message

    def __call__(self, query, **kwargs):
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query}
            ],
            **self.model_params
        }
        
        response = requests.post(self.endpoint, headers=headers, json=payload)
        
        if response.status_code == 200:
            try:
                output = response.json()
                response_text = output["choices"][0]["message"]["content"]
                return {"query": query, "response": response_text}
            except (KeyError, IndexError):
                return {"query": query, "response": "Error: Unexpected API response format"}
        else:
            return {"query": query, "response": f"Error: {response.status_code}, {response.text}"}
