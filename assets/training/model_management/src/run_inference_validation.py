# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the structure of expected and actual inference response JSON files."""

import json
import argparse
import os
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


def compare_structures(inference_payload, expected_response, inference_response, success_status, inference_time):
    """
    Compare JSON structures (keys only) of expected and actual.

    Returns a dictionary with validation results.
    """
    expected_structure = get_json_structure(expected_response)
    actual_structure = get_json_structure(inference_response)
    logger.info(f"expected_structure: {expected_structure} \n actual_structure: {actual_structure}")

    result = {
        "success": success_status,
        "inference_time" : inference_time,
        "sample_request": inference_payload,
        "sample_response": expected_response,
        "actual_response": inference_response,
        "structure_match": expected_structure == actual_structure if expected_response else None,
        "structural_difference": []
    }

    if not result["structure_match"]:
        result["differences"] = [
            {"expected": expected_structure, "actual": actual_structure}
        ]
    logger.info(f"validation result: {result}")
    return result


def save_validation_result(result, output_path):
    """Save validation results to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
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
        output_data_path = run_details['runDefinition']['outputData']['validation_result']['outputLocation']['uri']['path']
        
        output_data_uri = replace_name_in_path(output_data_path, run.id)

        # Extract datastore name and path from the AzureML URI
        datastore_name, path = extract_datastore_info(output_data_uri)
        
        # Construct the storage URI
        storage_uri = get_storage_url(datastore_name)
        full_storage_uri = f"{storage_uri}/{path}"
        logger.info(f"Full storage URI: {full_storage_uri}")
        
        return full_storage_uri
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

    args = parser.parse_args()

    # Load expected and actual responses.
    inference_payload = load_json_from_string(args.inference_payload)
    inference_output = load_json(args.inference_response)

    expected_response = load_json(args.expected_response) if args.expected_response else None
    logger.info(f"expected response: {expected_response}, actual response: {inference_response}")

    inference_response = inference_output.get("response")
    inference_time = inference_output.get("inference_time_ms", 0)  # Default to 0 if not present

    # Infer success status based on the presence of a valid response
    success_status = inference_response is not None and bool(inference_response)

    if expected_response:
        validation_result = validation_result = compare_structures(
            inference_payload,
            expected_response,
            inference_response,
            success_status,
            inference_time
        )
    else:
        validation_result = {
            "success": success_status,
            "inference_time": inference_time,
            "sample_request": inference_payload,
            "sample_response": expected_response,
            "actual_response": inference_response,
            "structure_match": None,
            "actual_structure": []
        }
        logger.info("No expected response provided. Skipping structure comparison.")

    # Save the validation result.
    save_validation_result(validation_result, args.validation_result)
    logger.info(f"validation_result: {validation_result}, Validation result saved to {args.validation_result}")

    store_metrics_paths(args.metrics_storage_uri)


if __name__ == "__main__":
    main()
