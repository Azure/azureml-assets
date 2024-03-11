# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""data upload component."""
import logging
from openai import AzureOpenAI
from azureml.core import Run

logger = logging.getLogger()

aoai_resource_name = "b-cc1c647c05a94c"
azure_endpoint = "https://swedencentral.api.cognitive.microsoft.com/"
api_version = "2023-12-01-preview"


def get_azure_oai_client():
    """Get azure oai client using api key from keyvault."""
    api_key_name = f"OPENAI-API-KEY-{aoai_resource_name}"
    run = Run.get_context()
    api_key = run.get_secret(api_key_name)
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version
    )
