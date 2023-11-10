# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for input/output operations."""

from typing import Any, List, Dict, Optional
import uuid
import json
import os
import glob

import mltable
from azureml._common._error_definition.azureml_error import AzureMLError

from .logging import get_logger
from .exceptions import DataFormatException
from .error_definitions import DataFormatError


logger = get_logger(__name__)


def _raise_if_not_jsonl_file(input_file_path: str) -> None:
    """Raise exception if file is not a .jsonl file.

    :param input_file_path: Path to file
    :return: None
    """
    if not input_file_path.endswith(".jsonl"):
        mssg = f"Input file '{input_file_path}' is not a .jsonl file."
        raise DataFormatException._with_error(
            AzureMLError.create(DataFormatError, error_details=mssg)
        )


def _is_mltable(dataset: str) -> bool:
    """
    Check if dataset is MLTable.

    :param dataset: Path to dataset
    :return: True if dataset is an MLTable, False otherwise
    """
    is_mltable = False
    if os.path.isdir(dataset):
        local_yaml_path = os.path.join(dataset, "MLTable")
        if os.path.exists(local_yaml_path):
            is_mltable = True
    return is_mltable


def _get_file_paths_from_folder(dataset: str) -> List[str]:
    """
    Get sorted file paths from folder.

    :param dataset: Path to dataset
    :return: List of sorted paths to files
    """
    file_paths = []

    for root, dirs, files in os.walk(dataset):
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_paths.sort()
    return file_paths


def get_output_file_path(input_file_path: str, output_path: str, counter: Optional[int] = None) -> str:
    """Get output file path.

    Uses the file name and extension from `input_file_path` along with counter to create the output file name.

    :param input_file_path: Path to input file
    :param output_path: Path to output directory
    :param counter: Counter for output file name
    :return: Path to output file
    """
    file_name, file_ext = os.path.splitext(os.path.basename(input_file_path))
    if counter is None:
        output_file_path = os.path.join(output_path, f"{file_name}{file_ext}")
    else:
        output_file_path = os.path.join(output_path, f"{file_name}_{counter}{file_ext}")
    return output_file_path


def resolve_io_path(dataset: str) -> List[str]:
    """Resolve input/output path as a list of file paths.

    It can handle the following cases for `dataset` argument:
    - `uri_file`: `dataset` is a single file.
    - `uri_folder`: `dataset` is a directory containing multiple files.
    - `mltable`: `dataset` is a directory containing an MLTable file.

    :param dataset: Either file or directory path
    :return: List of sorted file paths
    """
    if _is_mltable(dataset):
        logger.warning(
            "Received 'dataset' as MLTable. Trying to process."
        )
        df = mltable.load(dataset).to_pandas_dataframe()
        file_path = os.path.join(os.getcwd(), f"{uuid.uuid4()}.jsonl")
        df.to_json(file_path, orient="records", lines=True)
        return [file_path]

    if not os.path.isfile(dataset):
        logger.warning(
            "Received 'dataset' as URI_FOLDER. Trying to resolve paths."
        )
        return _get_file_paths_from_folder(dataset)

    return [dataset]


def read_jsonl_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Read `.jsonl` files and return a list of dictionaries.

    Ignores files that do not have `.jsonl` extension. Raises exception if no `.jsonl` files
    found or if any `.jsonl` file contains invalid JSON.

    :param file_paths: List of paths to .jsonl files.
    :return: List of dictionaries.
    """
    data_dicts = []
    for file_path in file_paths:
        if not file_path.endswith(".jsonl"):
            continue
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                try:
                    data_dicts.append(json.loads(line))
                except json.JSONDecodeError:
                    mssg = f"Invalid JSON format in line {i + 1} of file '{file_path}'."
                    raise DataFormatException._with_error(
                        AzureMLError.create(DataFormatError, error_details=mssg)
                    )
    if not data_dicts:
        mssg = "No .jsonl file found."
        raise DataFormatException._with_error(
            AzureMLError.create(DataFormatError, error_details=mssg)
        )
    return data_dicts


def read_json_data(data_path: Optional[str]) -> Dict[str, Any]:
    """
    Read json file(s) from a given file or directory path. \
    If multiple json files are found, they are merged in one dictionary.

    :param data_path: Path to the data.
    :returns: The dictionary representing the json data.
    """
    if data_path is None:
        return {}
    file_paths: List[str] = []
    if os.path.isfile(path=data_path):
        file_paths = [data_path]
    elif os.path.isdir(data_path):
        file_paths = glob.glob(
            pathname=os.path.join(data_path, "**/*.json"),
            recursive=True
        )
    json_output: Dict[str, Any] = {}
    for file_path in file_paths:
        with open(file=file_path, mode='r') as file:
            data: Dict[str, Any] = json.load(file)
        json_output = {**json_output, **data}

    # Read artifacts.
    file_paths = glob.glob(
        pathname=os.path.join(data_path, "*", "artifacts", "*"),
        recursive=True
    )
    for file_path in file_paths:
        try:
            logger.info(f"Trying to read metrics from {file_path}")
            with open(file=file_path, mode='r') as file:
                data: Dict[str, Any] = json.load(file)
            key = os.path.basename(file_path)
            json_output[key] = data
        except Exception as exception:
            logger.warning(f"Failed to read metrics from {file_path} due to {exception}")
    return json_output


def save_json_to_file(json_dict: Dict[Any, Any], path: str) -> None:
    """
    Save the json represented as dictionary as a file in the specified path.

    :param json_dict: The json represented as dictionary.
    :param path: The path where it has to be saved.
    """
    with open(path, mode='w') as file:
        json.dump(json_dict, file, indent=4)


def parse_jinja_template(template: str, data: Dict[str, Any]) -> str:
    """
    Parse a jinja template with the given arguments.

    :param template: The jinja template.
    :param kwargs: The arguments to be passed to the template.
    :returns: The parsed template.
    """
    from jinja2 import Template
    return Template(template).render(**data)


def write_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Write the data to a jsonl file.

    :param data: The data to be written.
    :param file_path: The path where the data has to be written.
    """
    with open(file_path, mode='w') as file:
        for row in data:
            file.write(json.dumps(row))
            file.write('\n')
