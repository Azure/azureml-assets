# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OAI Deployment."""

from typing import List

from openai import OpenAI, BadRequestError, APITimeoutError
from azureml._common._error_definition.azureml_error import AzureMLError

from .abstract_deployment import AbstractDeployment
from ...utils.helper import exponential_backoff
from ...utils.constants import Constants
from ...utils.logging import get_logger
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


class OAIDeployment(AbstractDeployment):
    """Class for OAI Deployment."""

    def __init__(
        self,
        deployment_name: str,
        api_key: str,
    ):
        """Initialize Deployment."""
        super().__init__()
        self.deployment_name = deployment_name
        self._client = OpenAI(
            api_key=api_key,
            max_retries=Constants.MAX_RETRIES_OAI,
            timeout=Constants.DEFAULT_HTTPX_TIMEOUT,
        )

    @exponential_backoff()
    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """
        Get embeddings for the given text.

        :param text: List of text to get embeddings for.
        :return: List of embeddings.
        """
        response = self._client.embeddings.create(
            input=text, model=self.deployment_name
        )
        return [embedding.embedding for embedding in response.data]

    def get_batch_size(self, longest_sentence: str, intital_batch_size: int) -> int:
        """
        Get the batch size that will fit the model context length.

        :param longest_sentence: Longest sentence in the input text.
        :param intital_batch_size: Initial batch size.
        :return: Batch size that fits the model context length.
        """
        _batch_size = intital_batch_size

        while _batch_size > 0:
            batch: List[str] = [longest_sentence] * _batch_size
            try:
                self._client.embeddings.create(
                    input=batch, model=self.deployment_name
                )
                logger.info(f"Using batch size {_batch_size}.")
                return _batch_size
            except (BadRequestError, APITimeoutError) as e:
                logger.info(
                    f"Reducing batch size from {_batch_size} to {_batch_size // 2} due to error: {e}"
                )
                _batch_size //= 2

        if _batch_size == 0:
            mssg = (
                "Longest sentence in your data does not fit the model context length. "
                "Either reduce the length of the sentences or use a model with larger context length."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
