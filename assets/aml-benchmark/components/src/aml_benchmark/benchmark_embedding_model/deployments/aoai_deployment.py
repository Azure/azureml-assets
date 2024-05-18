# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AOAI Deployment."""

from openai import AzureOpenAI

from .oai_deployment import OAIDeployment
from ..utils.constants import EmbeddingConstants
from ...utils.constants import Constants


class AOAIDeployment(OAIDeployment):
    """Class for AOAI Deployment."""

    def __init__(
        self,
        deployment_name: str,
        endpoint_url: str,
        api_key: str,
        api_version: str,
    ):
        """Initialize Deployment."""
        super().__init__(
            deployment_name=deployment_name,
            api_key=api_key,
        )
        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint_url,
            max_retries=Constants.MAX_RETRIES_OAI,
            timeout=EmbeddingConstants.DEFAULT_HTTPX_TIMEOUT,
        )
