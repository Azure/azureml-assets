# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Union, Any, List, Dict
import mltable
import uuid
import json
import os
import pandas as pd

from .logging import get_logger


logger = get_logger(__name__)


def _raise_if_not_jsonl_file(input_file_path: str) -> None:
    """Raise exception if file is not a .jsonl file.

    :param input_file_path: Path to file
    :type input_file_path: str
    :return: None
    :rtype: NoneType
    """
    if not input_file_path.endswith(".jsonl"):
        mssg = f"Input file '{input_file_path}' is not a .jsonl file."
        logger.error(mssg)
        raise ValueError(mssg)


def _is_mltable(dataset: str) -> bool:
    """
    Check if dataset is MLTable.

    :param dataset: Path to dataset
    :type dataset: str
    :return: True if dataset is an MLTable, False otherwise
    :rtype: bool
    """
    is_mltable = False
    if os.path.isdir(dataset):
        local_yaml_path = os.path.join(dataset, "MLTable")
        if os.path.exists(local_yaml_path):
            is_mltable = True
    return is_mltable


def _get_file_path_from_uri_folder(dataset: str) -> Union[str, None]:
    """
    Get file path from URI_FOLDER.

    :param dataset: Path to dataset
    :type dataset: str
    :return: Path to file
    :rtype: Unio[str, None]
    """
    file_path = None
    files = os.listdir(dataset)
    if len(files) == 1:
        file_path = os.path.join(dataset, files[0])
    return file_path


def resolve_io_path(dataset: str) -> str:
    """Resolve input/output path as a single file path.

    It can handle the following cases for `dataset` argument:
    - `uri_file`: `dataset` is a single file.
    - `uri_folder`: `dataset` is a directory containing a single file.
    - `mltable`: `dataset` is a directory containing an MLTable file.

    :param dataset: Either file or directory path
    :type dataset: str
    :return: Path to unique file in directory
    :rtype: str
    """
    if _is_mltable(dataset):
        logger.warn(
            "Received 'dataset' as MLTable instead of URI_FILE. Trying to process."
        )
        df = mltable.load(dataset).to_pandas_dataframe()
        file_path = os.path.join(dataset, f"{uuid.uuid4()}.jsonl")
        df.to_json(file_path, orient="records", lines=True)
        return file_path

    if not os.path.isfile(dataset):
        logger.warn(
            "Received 'dataset' as URI_FOLDER instead of URI_FILE. Trying to resolve file."
        )
        dataset = _get_file_path_from_uri_folder(dataset)
        if dataset is None:
            mssg = (
                "More than one file in URI_FOLDER. Please specify a .jsonl URI_FILE or a URI_FOLDER "
                "with a single .jsonl file or an MLTable in input 'dataset'."
            )
            logger.error(mssg)
            raise ValueError(mssg)
    _raise_if_not_jsonl_file(dataset)
    return dataset


def read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read a .jsonl file and return a list of dictionaries.

    Raises exception if file is not a .jsonl file or if the file contains invalid JSON.

    :param file_path: Path to .jsonl file.
    :return: List of dictionaries.
    """
    _raise_if_not_jsonl_file(file_path)
    data_dicts = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            try:
                data_dicts.append(json.loads(line))
            except json.JSONDecodeError:
                mssg = f"Invalid JSON format in line {i + 1} of file '{file_path}'."
                logger.error(mssg)
                raise ValueError(mssg)
    return data_dicts


def read_pandas_data(data_path: str) -> pd.DataFrame:
    """
    Read data that is formatted in a JSON line format and return a pandas dataframe. This supports URI_FOLDERs,
    if those folders only contain JSON line-formatted data. This does not require the files to have the .jsonl
    extension, however.

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
