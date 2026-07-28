# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for input/output operations."""

from typing import Any, List, Dict
import json
import os

from azureml.acft.common_components.utils.error_handling.exceptions import (
    ACFTValidationException,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ACFTUserError,
)
from azureml._common._error_definition.azureml_error import AzureMLError


def _filter_files_with_given_extension(
    file_paths: List[str], extension: str
) -> List[str]:
    """Filter and return the list of files with given extension."""
    return [file_path for file_path in file_paths if file_path.endswith(extension)]


def _get_file_paths_from_dir(dir: str) -> List[str]:
    """
    Get sorted file paths from directory.

    Args:
        dir (str): Directory path.

    Returns:
        file_paths (List[str]): List of sorted file paths.
    """
    file_paths = []

    for root, _, files in os.walk(dir):
        for file in files:
            file_paths.append(os.path.join(root, file))

    file_paths.sort()
    return file_paths


def _resolve_io_path(path: str) -> List[str]:
    """Resolve input/output path as a list of file paths.

    It can handle the following cases for `path` argument:
    - `uri_file`: `path` points to a single file.
    - `uri_folder`: `path` points to a directory containing multiple files.

    Args:
        path (str): Path to the file or folder.

    Returns:
        paths (List[str]): List of file paths.
    """
    if not os.path.isfile(path):
        return _get_file_paths_from_dir(path)

    return [path]


def read_jsonl_files(path: str) -> List[Dict[str, Any]]:
    """
    Read jsonl file/files and return a list of dictionaries.

    If `path` points to a file without extension, try to read it as \
    a jsonl file. This is done to support `uri_file` without extension scenario.

    Args:
        path (str): Path to a jsonl file or a directory containing jsonl files.

    Returns:
        data (List[Dict[str, Any]]): List of dictionaries.
    """
    file_paths: List[str] = _resolve_io_path(path)
    if len(file_paths) > 1:
        file_paths = _filter_files_with_given_extension(file_paths, ".jsonl")
    data_dicts = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for i, line in enumerate(file):
                try:
                    data_dicts.append(json.loads(line))
                except json.JSONDecodeError:
                    mssg = f"Invalid JSON format in line {i + 1} of file '{file_path}'."
                    raise ACFTValidationException._with_error(
                        AzureMLError.create(ACFTUserError, pii_safe_message=mssg)
                    )
    if not data_dicts:
        mssg = f"No data found in {file_paths}."
        raise ACFTValidationException._with_error(
            AzureMLError.create(ACFTUserError, pii_safe_message=mssg)
        )
    return data_dicts


def write_jsonl_file(file_path: str, data: List[Dict[str, Any]]) -> None:
    """Write data to a `.jsonl` file.

    Args:
        file_path (str): Path to the file.
        data (List[Dict[str, Any]]): Data to be written to the file.
    """
    with open(file_path, "w") as file:
        for line in data:
            file.write(json.dumps(line) + "\n")
