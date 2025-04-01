# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Update model onboarding version with CommonBench results."""

import sys
import os
import uuid
import json
import requests
import argparse
from datetime import datetime, timezone
from azure.identity import ManagedIdentityCredential
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azureml.model.mgmt.config import AppName
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.PUBHLISH_VALIDATION_RESULTS_SELF_SERVE


def read_results_from_file(file_path):
    """Read the metrics results from the given file path."""
    try:
        with open(file_path, 'r') as f:
            results_dict = json.load(f)
        print(f"Results loaded from {file_path}")
        return results_dict
    except Exception as e:
        print(f"Error reading from file: {e}")
        return None

def get_auth_token():
    """Generate auth token for Azure API."""
    is_obo = False
    tokenUri = "https://management.azure.com/.default"
    token = None

    try:
        credential = AzureMLOnBehalfOfCredential()
        token = credential.get_token(tokenUri).token
        is_obo = True
    except Exception:
        logger.warning(
            "Failed to get user credentials, fetching MSI credentials")

    if not is_obo:
        try:
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            token = credential.get_token(tokenUri).token
        except Exception as ex:
            raise Exception(f"Failed to get MSI credentials : {ex}")

    return token


def update_model_onboarding_version(
    publisher_name,
    model_name,
    model_version,
    sku,
    validation_id,
    selfserve_base_url,
    metrics_storage_uri
):
    """Update model onboarding version with benchmark results."""
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    metrics_path_dict = read_results_from_file(metrics_storage_uri)

    validationResultUrl = None

    if validation_id:
        if metrics_path_dict.get("api_inference_path"):
            validationResultUrl = metrics_path_dict.get("api_inference_path")
    else:
        logger.error(
            "Validation run ID is None, not updating validation results in self-serve")
        sys.exit(1)

    payload = {
        "passed": True,
        "status": "Completed",
        "message": "Validation Successful",
        "validationResult": validationResultUrl
    }

    api_url = f"{selfserve_base_url}/model-publisher-self-serve/publishers/{publisher_name}/models/{model_name}/versions/{model_version}/validations/{validation_id}/updateValidationResult?api-version=2024-12-31"

    headers = {
        "Authorization": f"Bearer {get_auth_token()}",
        "Content-Type": "application/json",
        "User-Agent": "AzureML-ModelPublishing/1.0"
    }

    try:
        logger.info(f"Sending request to {api_url} \n, headers: {headers} \n, payload: {payload}")

        response = requests.put(api_url, headers=headers, json=payload)

        logger.info(f"Response: {response.text}")

        if response.ok:
            logger.info(
                f"Successfully updated model onboarding version. Response: {response.status_code}")
            return {"status_code": response.status_code}
        else:
            logger.error(
                f"Failed to update model onboarding version. Status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise Exception(
                f"Request failed with status code {response.status_code}: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Update model onboarding version with CommonBench validation results")

    parser.add_argument("--publisher-name", required=True,
                        help="Name of the model publisher (e.g., ContosoAI)")
    parser.add_argument("--model-name", required=True,
                        help="Name of the model (e.g., VerboGenie)")
    parser.add_argument("--model-version", required=True,
                        help="Model onboarding version (e.g., 5)")
    parser.add_argument("--selfserve-base-url", required=True,
                        default="https://int.api.azureml-test.ms",
                        help="Base URL of the model publisher self-serve API")
    parser.add_argument("--validation-id", required=True,
                        help="Run ID of the validation run")
    parser.add_argument("--metrics-storage-uri", required=True,
                        help="URI to the storage where validation metrics are stored")
    parser.add_argument("--sku", required=False,
                        default="Standard_NC24ads_A100_v4",
                        help="Suggested SKU based on benchmark results")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    try:
        result = update_model_onboarding_version(
            args.publisher_name,
            args.model_name,
            args.model_version,
            args.sku,
            args.validation_id,
            args.selfserve_base_url,
            args.metrics_storage_uri
        )
        logger.info("Model onboarding version update completed successfully")
    except Exception as e:
        logger.error(f"Failed to update model onboarding version: {e}")
        sys.exit(1)
