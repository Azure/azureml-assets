# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OSS Deployment."""

from typing import List

from .deployment import Deployment


class OSSDeployment(Deployment):
    """Class for OSS Deployment."""

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
        raise NotImplementedError("oss deployment not supported at the moment.")

    def get_batch_size(self, longest_sentence: str, intital_batch_size: int) -> int:
        """
        Get the batch size that will fit the model context length.

        :param longest_sentence: Longest sentence in the input text.
        :param intital_batch_size: Initial batch size.
        :return: Batch size that fits the model context length.
        """
        raise NotImplementedError("oss deployment not supported at the moment.")

