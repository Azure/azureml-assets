import re
import os
import requests
import logging
from openai import AzureOpenAI
from azure.core.exceptions import HttpResponseError

API_VERSION = "2025-01-01-preview"
MAX_RETRIES = 0

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ModelTarget:
    def __init__(self, endpoint, api_key, deployment_name, model_params, system_message):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.model_params = model_params
        self.system_message = system_message

    def __call__(self, query, **kwargs):
        config = {
            "max_retries": MAX_RETRIES,
            "api_key": self.api_key,
            "api_version": API_VERSION,
            "azure_endpoint": self.endpoint,
        }

        client = AzureOpenAI(**config)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query}
        ]
        
        try:
            response = client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                **self.model_params
            )

            logger.info(f"  - Response: {response}")
            if response and response.choices:
                response_text = response.choices[0].message.content
                return {"query": query, "response": response_text}
            else:
                return {"query": query, "response": "Error: No choices returned in the response"}
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            return {"query": query, "response": f"Error: {str(e)}"}
