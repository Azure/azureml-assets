# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Update model onboarding version with CommonBench results."""

import sys
import os
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
    metrics_storage_uri,
    error_message
):
    """Update model onboarding version with benchmark results."""
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not metrics_storage_uri:
        validation_success = False
        metrics_url = None
    else:
        metrics_path_dict = read_results_from_file(metrics_storage_uri)
        metrics_url = metrics_path_dict.get(
                "api_inference_path") if metrics_path_dict else None
        validation_success = metrics_url is not None

    validation_result = []
    logger.info(f"validation_success: {validation_success}, metrics_url: {metrics_url}, metrics_storage_uri: {metrics_storage_uri}")

    if validation_id:
        validation_result.append({
                "Id": validation_id,
                "type": "API_VALIDATION",
                "passed": True,
                "message": "API inference passed successfully",
                "validationResultUrl": metrics_url,
                "errorMessage": error_message if error_message else None,
                "status": "Completed" if validation_success else "Failed",
                "createdTime": current_time,
                "updatedTime": current_time,
                "sku": sku
            })
    else:
        logger.error(
            "Validation ID is None, not updating validation results in self-serve")
        sys.exit(1)

    payload = {
        "passed": True,
        "status": "Completed",
        "message": "Validation Successful",
        "validationResult": validation_result
    }

    api_url = (
        f"{selfserve_base_url}/model-publisher-self-serve/publishers/{publisher_name}/models/{model_name}"
        f"/model-onboarding-version/{model_version}/updateModelOnboardingVersion?api-version=2024-12-31"
    )

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
    parser.add_argument("--metrics-storage-uri", required=False,
                        help="URI to the storage where validation metrics are stored")
    parser.add_argument("--sku", required=False,
                        default="Standard_NC24ads_A100_v4",
                        help="Suggested SKU based on benchmark results")
    parser.add_argument("--validation-error", required=False,
                        help="Path to the file containing validation error messages or stack traces")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")

    error_message = ""
    if args.validation_error:
        try:
            with open(args.validation_error, "r") as f:
                validation_error_message = f.read().strip()
                error_message += f"Validation Error: {validation_error_message}\n"
        except Exception as e:
            logger.warning(f"Failed to read validation_error file: {e}")

    try:
        result = update_model_onboarding_version(
            args.publisher_name,
            args.model_name,
            args.model_version,
            args.sku,
            args.validation_id,
            args.selfserve_base_url,
            args.metrics_storage_uri,
            error_message
        )
        logger.info("Model onboarding version update completed successfully")
    except Exception as e:
        logger.error(f"Failed to update model onboarding version: {e}")
        sys.exit(1)
