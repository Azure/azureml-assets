# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OSS Deployment."""

from typing import List, Optional
import requests
import json

from azureml._common._error_definition.azureml_error import AzureMLError

from .abstract_deployment import AbstractDeployment
from ...utils.helper import exponential_backoff
from ...utils.logging import get_logger
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


class OSSDeployment(AbstractDeployment):
    """Class for OSS Deployment."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        deployment_name: Optional[str] = None,
    ):
        """Initialize Deployment."""
        super().__init__()
        self.endpoint_url = endpoint_url
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": (api_key)
        }

    @exponential_backoff()
    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """
        Get embeddings for the given text.

        :param text: List of text to get embeddings for.
        :return: List of embeddings.
        """
        data = {
            "texts": text,
            "input_type": "search_document",
        }
        body = str.encode(json.dumps(data))
        response = requests.post(self.endpoint_url, data=body, headers=self._headers)
        response.raise_for_status()
        result = response.json()
        return result["embeddings"]

    @exponential_backoff()
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
            data = {
                "texts": batch,
                "input_type": "search_document",
            }
            body = str.encode(json.dumps(data))
            try:
                response = requests.post(self.endpoint_url, data=body, headers=self._headers)
                response.raise_for_status()
                logger.info(f"Using batch size {_batch_size}.")
                return _batch_size
            except requests.HTTPError as e:
                # 400 is returned by cohere endpoint when the batch size is too large
                if e.response.status_code != 400:
                    raise
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
