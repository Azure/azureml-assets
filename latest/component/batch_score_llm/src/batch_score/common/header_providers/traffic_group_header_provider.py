# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for traffic group header provider."""

from .header_provider import HeaderProvider
from .. import constants


class TrafficGroupHeaderProvider(HeaderProvider):
    """Traffic group header provider."""

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        return {
            'azureml-model-group': constants.TRAFFIC_GROUP,
            'azureml-collect-request': 'false',
            'azureml-inferencing-offer-name': 'azureml_vanilla',
        }
