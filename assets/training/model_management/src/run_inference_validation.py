# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the structure of expected and actual inference response JSON files."""

import base64
import json
import argparse
import os
import sys
from datetime import datetime, timezone
from azureml.core import Run
from azureml.model.mgmt.utils.common_utils import get_mlclient
from azureml.model.mgmt.config import AppName
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.RUN_INFERENCE_VALIDATION


def load_json(file_path):
    """Load JSON data from a file. If the loaded data is a string, try to parse it as JSON."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        # If data is a string, parse it as JSON.
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                logger.warning(f"Error parsing JSON from string in {file_path}: {e}")
        return data
    except Exception as e:
        logger.warning(f"Error loading JSON file {file_path}: {e}")
        return None


def load_json_from_string(json_string):
    """Load JSON data from a string."""
    try:
        data = json.loads(json_string)
        return data
    except Exception as e:
        logger.warning(f"Error parsing JSON from string: {e}")
        return None


def get_json_structure(data):
    """
    Recursively extract the structure of JSON (keys only).

    For dictionaries, returns a dict of keys mapped to their structure.
    For lists, returns a list with the structure of the first element.
    For other types, returns None.
    """
    if isinstance(data, dict):
        return {key: get_json_structure(value) for key, value in data.items()}
    elif isinstance(data, list) and len(data) > 0:
        # Assume all elements share the same structure and return the structure of the first element.
        return [get_json_structure(data[0])]
    else:
        return None


def compare_structures(expected_response, actual_response):
    """
    Compare JSON structures (keys only) of expected and actual.

    Returns a dictionary with structural differences and a match flag.
    """
    expected_structure = get_json_structure(expected_response)
    actual_structure = get_json_structure(actual_response)
    logger.info(f"expected_structure: {expected_structure} \n actual_structure: {actual_structure}")

    structure_match = expected_structure == actual_structure if expected_response else None
    structural_difference = []

    if not structure_match:
        structural_difference = [
            {"expected": expected_structure, "actual": actual_structure}
        ]

    logger.info(f"Structure match: {structure_match}, Structural differences: {structural_difference}")
    return {
        "structure_match": structure_match,
        "structural_difference": structural_difference
    }


def save_validation_result(request_details, output_dir, validation_id, sku, status):
    """Save validation results to a JSON file."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "validation_result.json")

        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        validation_result = {
            "id": validation_id,
            "sku": sku,
            "createdTime": current_time,
            "updatedTime": current_time,
            "type": "MAAP_INFERENCING",
            "status": status,
            "requestDetails": request_details
        }

        with open(output_path, "w") as f:
            json.dump(validation_result, f, indent=4)
        logger.info(f"Validation result saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation result: {e}")


def replace_name_in_path(path_template, name_value):
    """Replace the placeholder in the output path with the actual job name."""
    return path_template.replace('${{name}}', name_value)


def fetch_storage_uri():
    """Return the storage URI of the output file from the AzureML pipeline run."""
    try:
        run = Run.get_context()
        run_details = run.get_details()
        output_data = run_details['runDefinition']['outputData']['validation_result']['outputLocation']['uri']
        output_data_path = output_data['path']

        output_data_uri = replace_name_in_path(output_data_path, run.id)
        # Extract datastore name and path from the AzureML URI
        datastore_name, path = extract_datastore_info(output_data_uri)

        # Construct the storage URI
        storage_uri = get_storage_url(datastore_name)
        folder_uri = f"{storage_uri}/{path}"
        # Construct the full path to the validation_result.json file
        full_file_uri = f"{folder_uri}/validation_result.json"

        logger.info(f"Full storage URI (file): {full_file_uri}")

        return full_file_uri  # This is the full path to validation_result.json
    except Exception as e:
        logger.error(f"Error fetching storage URI: {e}")
        return None


def store_metrics_paths(metrics_file_path):
    """Store the paths of the metrics CSV files in a JSON file."""
    base_path = fetch_storage_uri()

    logger.info(f"validation_result_path: {base_path}")
    result_dict = {}
    result_dict['api_inference_path'] = base_path
    if result_dict:
        write_results_to_file(result_dict, metrics_file_path)


def fetch_path(output_dir):
    """Return the relative path of the data from the output directory."""
    try:
        # Calculate relative path from the job folder
        rel_path = os.path.relpath(output_dir, os.getcwd())
        logger.info(f"api inference validation relative path: {rel_path}")
        result_dict = {
            'api_inference_path': rel_path
        }
        return result_dict
    except Exception as e:
        logger.error(f"Error calculating relative path: {e}")
        return {}


def write_results_to_file(results_dict, file_path):
    """Write the results dictionary to a JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        logger.info(f"Results written to {file_path} in JSON format")
        return True
    except Exception as e:
        logger.error(f"Error writing to file: {e}")
        return False


def get_storage_url(datastore_name):
    """Retrieve the storage URL for the specified datastore."""
    # Get MLClient instance
    ml_client = get_mlclient()
    datastore = ml_client.datastores.get(datastore_name)
    storage_account_name = datastore.account_name
    container_name = datastore.container_name
    endpoint = datastore.endpoint

    storage_uri = f"https://{storage_account_name}.blob.{endpoint}/{container_name}"
    logger.info(f"validation result storage: {storage_uri}")

    return storage_uri


def extract_datastore_info(datastore_uri_path):
    """Extract both datastore name and path from an Azure ML datastore URI path."""
    # Check if it's a valid datastore URI
    if not datastore_uri_path.startswith('azureml://datastores/'):
        return None, None

    parts = datastore_uri_path.split('/')

    # The datastore name should be the part after 'datastores/'
    if len(parts) >= 5 and parts[0] == 'azureml:' and parts[1] == '' and parts[2] == 'datastores' and 'paths' in parts:
        datastore_name = parts[3]

        # Find the index of 'paths' in the URI
        paths_index = parts.index('paths')

        # Join everything after 'paths/' to form the path
        path = '/'.join(parts[(paths_index + 1):])

        return datastore_name, path

    return None, None


def main():
    """Compare expected and actual inference response structures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_payload", type=str, required=True,
                        help="Serialized JSON payload for inference")
    parser.add_argument("--expected_response", type=str, required=False,
                        help="Path to the expected inference response JSON file.")
    parser.add_argument("--inference_response", type=str, required=True,
                        help="Path to the actual inference response JSON file.")
    parser.add_argument("--validation_result", type=str, required=True,
                        help="Path to save validation results.")
    parser.add_argument("--metrics_storage_uri", type=str, required=True,
                        help="Path to store the metrics.")
    parser.add_argument("--sku", required=False,
                        default="Standard_NC24ads_A100_v4",
                        help="Suggested SKU based on benchmark results")
    parser.add_argument("--validation-id", required=True,
                        help="Run ID of the validation run")

    args = parser.parse_args()

    inference_payload = None
    if args.inference_payload:
        decoded_bytes = base64.b64decode(args.inference_payload)

        # Convert bytes to string
        decoded_str = decoded_bytes.decode('utf-8')
        logger.info(f"Decoded string: {decoded_str}")

        inference_payload = json.loads(decoded_str)

    expected_response = None
    if args.expected_response:
        decoded_bytes = base64.b64decode(args.expected_response)

        # Convert bytes to string
        decoded_str = decoded_bytes.decode('utf-8')
        logger.info(f"Decoded string: {decoded_str}")
        expected_response = json.loads(decoded_str)


    inference_output = load_json(args.inference_response)
    if not inference_output:
        logger.error("Inference output is missing or invalid.")
        sys.exit(1)

    inference_response = inference_output.get("response")
    if isinstance(inference_response, str):
        try:
            inference_response = json.loads(inference_response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse actualResponse as JSON: {e}")

    if inference_response is None:
        logger.warning("Actual response is missing or invalid. Setting it to an empty structure.")
        inference_response = {}

    inference_time = inference_output.get("inference_time", 0)
    logger.info(f"inference_payload: {inference_payload}, expected response: {expected_response}, "
                f"actual response: {inference_response}")

    # Infer success status based on the presence of a valid response
    success_status = inference_response is not None and bool(inference_response)
    status = "Success" if success_status else "Failed"

    request_details = {
        "providedRequest": inference_payload,
        "providedResponse": expected_response,
        "actualResponse": inference_response,
        "responseTimeMs": inference_time,
        "errorMessage": None,
        "structuralDiff": None,
    }
    if expected_response:
        comparison_result = compare_structures(expected_response, inference_response)
        request_details["structuralDiff"] = comparison_result.get("structural_difference", [])

    # Save the validation result.
    save_validation_result(request_details, args.validation_result, args.validation_id, args.sku, status)
    logger.info(f"validation_result: {request_details}, Validation result saved to {args.validation_result}")

    store_metrics_paths(args.metrics_storage_uri)


if __name__ == "__main__":
    main()
