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
from datetime import datetime
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


def update_model_onboarding_version(
    publisher_name,
    model_name,
    model_version,
    selfserve_base_url,
    sku,
    metrics_storage_uri
):
    """Update model onboarding version with benchmark results."""
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    is_obo = False
    try:
        credential = AzureMLOnBehalfOfCredential()
        token = credential.get_token(
            "https://management.azure.com/.default").token
        is_obo = True
    except Exception as ex:
        logger.warning(f"Failed to get OBO credentials - {ex}")

    if not is_obo:
        try:
            logger.info("Fetching MSI credential")
            msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            credential = ManagedIdentityCredential(client_id=msi_client_id)
            token = credential.get_token(
                "https://management.azure.com/.default").token
        except Exception as ex:
            raise (f"Failed to get MSI credentials : {ex}")

    metrics_path_dict = read_results_from_file(metrics_storage_uri)

    run_id = str(uuid.uuid4())

    validation_result = []

    if metrics_path_dict.get("perf_bench_path") is not None:
        validation_result.append({
            "runId": run_id,
            "type": "PERF_BENCHMARK",
            "passed": True,
            "message": "Baseline data is captured successfully",
            "validationResultUrl": metrics_path_dict.get("perf_bench_path"),
            "createdTime": current_time,
            "status": "success",
            "sku": sku
        })

    if metrics_path_dict.get("api_validation_path") is not None:
        validation_result.append({
            "runId": run_id,
            "type": "API_VALIDATION",
            "passed": True,
            "message": "API validation passed successfully",
            "validationResultUrl": metrics_path_dict.get("api_validation_path"),
            "status": "success",
            "createdTime": current_time,
            "sku": sku
        })

    if metrics_path_dict.get("api_inference_path") is not None:
        validation_result.append({
            "runId": run_id,
            "type": "API_VALIDATION",
            "passed": True,
            "message": "API inference passed successfully",
            "validationResultUrl": metrics_path_dict.get("api_inference_path"),
            "status": "success",
            "createdTime": current_time,
            "sku": sku
        })

    payload = {
        "suggestedSKU": sku,
        "status": "Validation",
        "subStatus": "Validation_Successful",
        "validationResult": validation_result
    }

    api_url = f"{selfserve_base_url}/model-publisher-self-serve/publishers/{publisher_name}/models/{model_name}/model-onboarding-version/{model_version}/updateModelOnboardingVersion?api-version=2024-12-31"

    headers = {
        "Authorization": f"Bearer {token}",
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
            args.selfserve_base_url,
            args.sku,
            args.metrics_storage_uri
        )
        logger.info("Model onboarding version update completed successfully")
    except Exception as e:
        logger.error(f"Failed to update model onboarding version: {e}")
        sys.exit(1)
