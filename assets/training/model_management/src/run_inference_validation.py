# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the structure of expected and actual inference response JSON files."""

import base64
import json
import argparse
import os
import sys
import traceback
import re
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


def set_nested_value(d, keys, value):
    """
    Helper to set a value into a nested dictionary/list from a list of keys/indexes.
    """
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        if isinstance(key, int):
            while len(d) <= key:
                d.append({} if not is_last else None)
            if is_last:
                d[key] = value
            else:
                if not isinstance(d[key], (dict, list)):
                    d[key] = {}
                d = d[key]
        else:
            if key not in d or not isinstance(d[key], (dict, list)):
                d[key] = {} if not is_last else None
            if is_last:
                d[key] = value
            else:
                d = d[key]

def parse_key_path(key):
    """
    Converts a key string like '[0].a.b[1]' to a list of keys: [0, 'a', 'b', 1]
    """
    parts = re.findall(r'\[(\d+)\]|([^.]+)', key)
    return [int(i) if i else j for i, j in parts]

def build_nested_json(flat_dict):
    """
    Converts a flat key-path dictionary to nested JSON.
    """
    result = {} if flat_dict else None
    for key_path, value in flat_dict.items():
        keys = parse_key_path(key_path)
        if isinstance(keys[0], int):
            if not isinstance(result, list):
                result = []
        set_nested_value(result, keys, value)
    return result

def get_json_structure_with_values(data, parent_key=''):
    """
    Recursively extract key paths and their values from nested JSON.
    Returns a dictionary of full_key_path: value
    """
    items = {}
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, (dict, list)):
                items.update(get_json_structure_with_values(v, full_key))
            else:
                items[full_key] = v
    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{parent_key}[{index}]" if parent_key else f"[{index}]"
            if isinstance(item, (dict, list)):
                items.update(get_json_structure_with_values(item, full_key))
            else:
                items[full_key] = item
    return items

def compare_structures(expected_response, actual_response):
    """
    Compare JSON structures and return full nested added/removed diffs.
    """
    expected_structure = get_json_structure_with_values(expected_response)
    actual_structure = get_json_structure_with_values(actual_response)

    logger.info(f"Expected flat structure: {expected_structure}")
    logger.info(f"Actual flat structure: {actual_structure}")

    added_keys = actual_structure.keys() - expected_structure.keys()
    removed_keys = expected_structure.keys() - actual_structure.keys()

    added_flat = {key: actual_structure[key] for key in added_keys}
    removed_flat = {key: expected_structure[key] for key in removed_keys}

    added_nested = build_nested_json(added_flat)
    removed_nested = build_nested_json(removed_flat)

    structure_match = not added_flat and not removed_flat

    result = {
        "structure_match": structure_match,
        "structural_difference": {
            "added": added_nested,
            "removed": removed_nested
        }
    }

    logger.info("Comparison result:")
    logger.info(json.dumps(result, indent=4))

    return result

def save_validation_result(request_details, output_dir, validation_id, sku, status):
    """Save validation results to a JSON file."""
    try:
        logger.info(f"Saving validation result to {output_dir}")
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        output_path = os.path.join(output_dir, "validation_result.json")
        logger.info(f"Output path: {output_path}")

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
        raise Exception(f"Failed to get MSI credentials : {e}")


def replace_name_in_path(path_template, name_value):
    """Replace the placeholder in the output path with the actual job name."""
    return path_template.replace('${{name}}', name_value)


def fetch_storage_uri():
    """Return the storage URI of the output file from the AzureML pipeline run."""
    try:
        run = Run.get_context()
        run_details = run.get_details()
        output_data = run_details['runDefinition']['outputData']['validation_results']['outputLocation']['uri']
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


def run_inference_validation():
    """Perform the inference validation logic."""
    try:
        args = parse_args()
        error_message = ""
        if args.deployment_error:
            try:
                with open(args.deployment_error, "r") as f:
                    deployment_error = f.read().strip()
                    error_message += deployment_error
            except Exception as e:
                logger.warning(f"Failed to read validation_error file: {e}")

        if args.validation_error:
            with open(args.validation_error, "w") as error_file:
                error_file.write(error_message)
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

        inference_output = None
        if args.inference_response:
            inference_output = load_json(args.inference_response)
            if not inference_output:
                logger.error("Inference response is missing or invalid.")

        inference_response = None
        if inference_output:
            inference_response = inference_output.get("response")
            if isinstance(inference_response, str):
                try:
                    inference_response = json.loads(inference_response)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse actualResponse as JSON: {e}")

        if inference_response is None:
            logger.warning("Actual response is missing or invalid. Setting it to an empty structure.")
            inference_response = {}

        inference_time = inference_output.get("inference_time", 0) if inference_output else 0
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
            "errorMessage": error_message,
            "structuralDiff": None,
        }
        logger.info(f"Request details: {request_details}")
        if expected_response and inference_response:
            comparison_result = compare_structures(expected_response, inference_response)
            request_details["structuralDiff"] = comparison_result.get("structural_difference", [])

        # Save the validation result.
        save_validation_result(request_details, args.validation_results, args.validation_id, args.sku, status)
        logger.info(f"validation_result: {request_details}, Validation result saved to {args.validation_results}")

        store_metrics_paths(args.metrics_storage_uri)
    except Exception as e:
        stack_trace = traceback.format_exc()
        error_message = f"Model validation failed.\n{stack_trace}"
        logger.error(error_message)
        # Save the error message in the request details
        request_details = {
            "providedRequest": None,
            "providedResponse": None,
            "actualResponse": None,
            "responseTimeMs": 0,
            "errorMessage": error_message,
            "structuralDiff": None,
        }

        # Save the validation result with the error message
        save_validation_result(request_details, args.validation_results, args.validation_id, args.sku, "Failed")

        # Write the error message to the specified error output file
        if args.validation_error:
            with open(args.validation_error, "w") as error_file:
                error_file.write(error_message)
        # raise Exception(f"Failed to run inference validation: {error_message}")

def main():
    run_inference_validation()


def parse_args():
    """Compare expected and actual inference response structures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_payload", type=str, required=True,
                        help="Serialized JSON payload for inference")
    parser.add_argument("--expected_response", type=str, required=False,
                        help="Path to the expected inference response JSON file.")
    parser.add_argument("--inference_response", type=str, required=False,
                        help="Path to the actual inference response JSON file.")
    parser.add_argument("--deployment_error", type=str, required=False,
                        help="Path to the deployment_error.")
    parser.add_argument("--validation_results", type=str, required=True,
                        help="Path to save validation results.")
    parser.add_argument("--metrics_storage_uri", type=str, required=True,
                        help="Path to store the metrics.")
    parser.add_argument("--sku", required=False,
                        default="Standard_NC24ads_A100_v4",
                        help="Suggested SKU based on benchmark results")
    parser.add_argument("--validation-id", required=True,
                        help="Run ID of the validation run")
    parser.add_argument("--validation_error", type=str, required=False,
                        help="Path to the file where error messages or stack traces will be written.")

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    return args


if __name__ == "__main__":
    main()
