# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OSS MaaP Deployment."""

from typing import List

from azureml._common._error_definition.azureml_error import AzureMLError

from .abstract_deployment import AbstractDeployment
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


class OSSMaaPDeployment(AbstractDeployment):
    """Class for OSS MaaP Deployment."""

    def __init__(
        self,
        deployment_name: str,
        endpoint_url: str,
        api_key: str,
    ):
        """Initialize Deployment."""
        self.model = deployment_name
        self.endpoint_url = endpoint_url

    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """
        Get embeddings for the given text.

        :param text: List of text to get embeddings for.
        :return: List of embeddings.
        """
        mssg = "oss maap deployment not supported at the moment."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    def get_batch_size(self, longest_sentence: str, intital_batch_size: int) -> int:
        """
        Get the batch size that will fit the model context length.

        :param longest_sentence: Longest sentence in the input text.
        :param intital_batch_size: Initial batch size.
        :return: Batch size that fits the model context length.
        """
        mssg = "oss maap deployment not supported at the moment."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )
