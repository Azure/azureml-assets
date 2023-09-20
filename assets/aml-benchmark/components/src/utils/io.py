# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for input/output operations."""

from typing import Any, List, Dict, Optional
import uuid
import json
import os
import pandas as pd

import mltable
from azureml._common._error_definition.azureml_error import AzureMLError

from utils.logging import get_logger
from utils.exceptions import DataFormatException
from utils.error_definitions import DataFormatError


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
        logger.warn(
            "Received 'dataset' as MLTable. Trying to process."
        )
        df = mltable.load(dataset).to_pandas_dataframe()
        file_path = os.path.join(dataset, f"{uuid.uuid4()}.jsonl")
        df.to_json(file_path, orient="records", lines=True)
        return [file_path]

    if not os.path.isfile(dataset):
        logger.warn(
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


def read_pandas_data(data_path: str) -> pd.DataFrame:
    """
    Read data that is formatted in a JSON line format and return a pandas dataframe.

    This supports URI_FOLDERs, if those folders only contain JSON line-formatted data.
    This does not require the files to have the .jsonl extension, however.

    Raises exception if the file contains invalid JSON.

    :param file_path: Path to JSON file/folder.
    :return: Pandas Dataframe
    """
    all_data = pd.DataFrame()

    # Check whether we're working with URI_FILE, URI_FOLDER, or MLTABLE
    if _is_mltable(data_path):
        logger.info("Received MLTABLE; assuming all files in the MLTABLE contain jsonl-formatted data.")
        try:
            all_data = mltable.load(data_path).to_pandas_dataframe()
        except json.JSONDecodeError:
            mssg = f"Invalid JSON format in provided file given by the MLTABLE '{data_path}'."
            logger.error(mssg)
            raise ValueError(mssg)

    elif os.path.isdir(data_path):
        logger.info("Received URI_FOLDER; assuming all files contain jsonl-formatted data.")
        for f in os.listdir(data_path):
            try:
                df = pd.read_json(os.path.join(data_path, f), lines=True)
                all_data = pd.concat([all_data, df])
            except json.JSONDecodeError:
                mssg = f"Invalid JSON format in the file '{os.path.join(data_path, f)}'."
                logger.error(mssg)
                raise ValueError(mssg)

    else:
        logger.info("Received URI_FILE; assuming the file contains jsonl-formatted data.")
        try:
            all_data = pd.read_json(data_path, lines=True)
        except json.JSONDecodeError:
            mssg = f"Invalid JSON format in the file '{data_path}'."
            logger.error(mssg)
            raise ValueError(mssg)

    return all_data
