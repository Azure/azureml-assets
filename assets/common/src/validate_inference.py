# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the structure of expected and actual inference response JSON files."""

import json
import argparse
from utils.config import AppName
from utils.logging_utils import custom_dimensions, get_logger


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.VALIDATE_INFERENCE


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

def compare_structures(expected, actual):
    """
    Compare JSON structures (keys only) of expected and actual.
    Returns a dictionary with validation results.
    """
    expected_structure = get_json_structure(expected)
    actual_structure = get_json_structure(actual)
    logger.info(f"expected_structure: {expected_structure} \n actual_structure: {actual_structure}")

    result = {
        "structure_match": expected_structure == actual_structure,
        "expected_structure": expected_structure,
        "actual_structure": actual_structure,
        "differences": []
    }

    if not result["structure_match"]:
        result["differences"] = [
            {"expected": expected_structure, "actual": actual_structure}
        ]
    logger.info(f"result: {result}")
    return result

def save_validation_result(result, output_path):
    """Save validation results to a JSON file."""
    try:
        with open(output_path, "w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Validation result saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving validation result: {e}")

def main():
    """Main function to compare expected and actual inference response structures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected_response", type=str, required=True,
                        help="Path to the expected inference response JSON file.")
    parser.add_argument("--actual_response", type=str, required=True,
                        help="Path to the actual inference response JSON file.")
    parser.add_argument("--validation_result", type=str, required=True,
                        help="Path to save validation results.")

    args = parser.parse_args()

    # Load expected and actual responses.
    expected = load_json(args.expected_response)
    actual = load_json(args.actual_response)
    logger.info(f"expected response: {expected}, actual response: {actual}")

    if expected is None or actual is None:
        logger.warning("One or both JSON files could not be loaded.")
        return

    # Compare the JSON structures.
    validation_result = compare_structures(expected, actual)

    # Save the validation result.
    save_validation_result(validation_result, args.validation_result)

if __name__ == "__main__":
    main()
