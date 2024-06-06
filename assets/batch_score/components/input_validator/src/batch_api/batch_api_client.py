# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch API Client."""

import requests
from azure.identity import DefaultAzureCredential
from logging import getLogger

from batch_api.data_validation_result import DataValidationResult

BATCH_DATA_VALIDATION_RESULT_SUBMISSION_URL = "https://managed-batch-inference"   # TODO: Update the URL

logger = getLogger(__name__)


class BatchApiClient:
    """Client for interacting with the MBI service."""

    def __init__(self) -> None:
        """Initializes the BatchApiClient."""
        self._credential = DefaultAzureCredential()

    def submit_validation_result(self, data_validation_result: DataValidationResult) -> None:
        """Submit the results of the data validation to the MBI service."""
        logger.info("Submitting the data validation result to the MBI service")

        auth_token = self._credential.get_token(BATCH_DATA_VALIDATION_RESULT_SUBMISSION_URL).token
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }

        response = requests.put(
            BATCH_DATA_VALIDATION_RESULT_SUBMISSION_URL,
            headers=headers,
            json=data_validation_result
        )

        logger.info(f"MBI service response: Status Code: '{response.status_code}', Response:  '{response.text}'")
