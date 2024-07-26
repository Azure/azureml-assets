# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa: E702

"""Model Prediction Data utilities."""
import sys

sys.path.append("/src/")

import local_constants
import llm.optimized.inference.constants as constants
import numpy as np
import pandas as pd
import os
import json
import glob

from mltable import load
from exceptions import DataLoaderException
from error_definitions import BadLabelColumnData
from logging_utilities import get_logger, get_azureml_exception, log_traceback

logger = get_logger(name=__name__)

def check_and_return_if_mltable(data):
    """Is current director MLTable or not.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_mltable = False
    if os.path.isdir(data):
        local_yaml_path = os.path.join(data, local_constants.MLTABLE_FILE_NAME)
        if os.path.exists(local_yaml_path):
            is_mltable = True
    return is_mltable


def read_model_prediction_data(file_path, batch_size=None, nrows=None):
    """Util function for reading test data for model prediction.

    Args:
        file_path (_type_): _description_
        task (_type_): _description_
        batch_size (_type_): _description_
        nrows (_type_): _description_

    Raises:
        DataValidationException: _description_

    Returns:
        _type_: _description_
    """
    logger.info(f"Batch Size: {batch_size}, Type: {type(batch_size)}")
    data = read_data(file_path, batch_size, nrows)
    return data


def read_data(file_path, batch_size=None, nrows=None):
    """Util function for reading test data.

    Args:
        file_path (_type_): _description_
        batch_size (_type_): _description_
        nrows (_type_): _description_

    Raises:
        DataValidationException: _description_

    Returns:
        _type_: _description_
    """
    is_mltable = check_and_return_if_mltable(file_path)
    if not is_mltable and os.path.isdir(file_path):
        logger.warning("Received URI_FOLDER instead of URI_FILE. Checking if part of LLM Pipeline")
        if local_constants.LLM_FT_PREPROCESS_FILENAME not in os.listdir(file_path):
            message = "Test Data is a folder. JSON Lines File or MLTable expected."
            raise ValueError(message)
        logger.info("Found LLM Preprocess args")
        with open(os.path.join(file_path, local_constants.LLM_FT_PREPROCESS_FILENAME)) as f:
            llm_preprocess_args = json.load(f)
        test_data_path = llm_preprocess_args[local_constants.LLM_FT_TEST_DATA_KEY]
        file_path = os.path.join(file_path, test_data_path)
    try:
        if is_mltable:
            if batch_size:
                logger.warning(
                    "batch_size not supported with MLTable files. Ignoring parameter."
                )
            mltable_data = load(file_path)
            if nrows:
                mltable_data = mltable_data.take(nrows)
            data = iter([mltable_data.to_pandas_dataframe()])
        else:
            data = read_dataframe(file_path, batch_size=batch_size, nrows=nrows)
            if not batch_size:
                data = iter([data])
    except Exception as e:
        raise e
    return data


def read_dataframe(file_path, batch_size=None, nrows=None):
    """Util function for reading a DataFrame based on the file extension.

    Args:
        file_path (_type_): _description_
        batch_size: (_type_, optional): _description_. Defaults to None.
        nrows (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.info("Detected File Format: {}".format(file_extension))
    if batch_size:
        nrows = None

    if file_extension == '.csv':
        # Reading a CSV file with the specified batch_size
        return pd.read_csv(file_path, chunksize=batch_size, nrows=nrows)
    elif file_extension == '.tsv':
        # Reading a TSV file with the specified batch_size and skipping initial spaces
        return pd.read_csv(file_path, sep='\t', chunksize=batch_size, nrows=nrows, skipinitialspace=True)
    elif file_extension == '.json':
        try:
            if batch_size:
                logger.warning("batch_size not supported for json file format. Ignoring parameter.")
            json_data = pd.read_json(file_path)
            return iter([json_data]) if batch_size else json_data
        except Exception as e:
            logger.error("Failed to load data json data. Exception: {}".format(str(e)))
            logger.info("Trying to load the data with 'lines=True'.")
            # Reading a JSONL file with the specified batch_size
            return pd.read_json(file_path, lines=True, dtype=False, chunksize=batch_size, nrows=nrows)
    elif file_extension == '.jsonl':
        try:
            # Reading a JSONL file with the specified batch_size
            return pd.read_json(file_path, lines=True, dtype=False, chunksize=batch_size, nrows=nrows)
        except Exception as e:
            logger.error("Failed to load data with JSONL. Trying to load the data without 'lines=True'. "
                         "Exception: {}".format(str(e)))
            json_data = pd.read_json(file_path)
            return iter([json_data]) if batch_size else json_data
    else:
        # Default to reading JSONL files without raising an exception
        if file_extension == "":
            logger.info("No file format detected. Loading as 'jsonl' file format.")
        else:
            logger.warning("File format not in supported formats. Defaulting to load data as jsonl format. "
                           "Valid formats: csv, tsv, json, jsonl.")
        return pd.read_json(file_path, lines=True, dtype=False, chunksize=batch_size, nrows=nrows)


def read_multiple_files(path):
    """Read multiple JSON Lines file from folder.

    Args:
        path (_type_): _description_

    Raises:
        DataLoaderException: _description_

    Returns:
        _type_: _description_
    """
    dfs = []
    for file_path in glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True):
        df = read_data(file_path=file_path)
        df = list(df)[0]
        dfs.append(df)
    if not dfs:
        error = "No JSON Lines files found in folder."
        raise ValueError(error)
    data = pd.concat(dfs, ignore_index=True)
    return iter([data])


def prepare_chat_data_from_ft_pipeline(data: pd.DataFrame):
    """Prepare Chat completion data from FT pipeline.
    Args:
        data: pd.DataFrame
    """
    try:
        messages_col = data[local_constants.LLM_FT_CHAT_COMPLETION_KEY]
    except Exception as e:
        logger.error(f"'{local_constants.LLM_FT_CHAT_COMPLETION_KEY}' not found in FT test dataset.")
        exception = get_azureml_exception(DataLoaderException, BadLabelColumnData, e, error=repr(e))
        log_traceback(exception, logger)
        raise exception
    X_test, y_test = {local_constants.LLM_FT_CHAT_COMPLETION_KEY:[]}, []
    for message in messages_col.to_list():
        X_test[local_constants.LLM_FT_CHAT_COMPLETION_KEY].append(message[:-1])
        y_test.append(message[-1]["content"])
    X_test = pd.DataFrame(X_test)
    y_test = pd.Series(y_test)
    return X_test, y_test.values


def prepare_data(data, task, label_column_name=None, _has_multiple_output=False, extra_y_test_cols=None):
    """Prepare data.

    Args:
        data (_type_): _description_
        task (_type_): _description_
        label_column_name (_type_, optional): _description_. Defaults to None.
        _has_multiple_output (bool, optional): _description_. Defaults to False.
        extra_y_test_cols (_type_, optional): _description_. Defaults to None.

    Raises:
        ModelEvaluationException: _description_
        DataLoaderException: _description_

    Returns:
        _type_: _description_
    """
    X_test, y_test = data, None
    if extra_y_test_cols is not None and label_column_name is not None:
        # IF extra_y_test_cols is not None, label_column_name should also be not None;
        # extra_y_test_cols is accepted only for text-gen
        X_test, y_test = data.drop(extra_y_test_cols + [label_column_name], axis=1), \
                         data[extra_y_test_cols + [label_column_name]]
    elif label_column_name is not None:
        X_test, y_test = data.drop(label_column_name, axis=1), data[label_column_name]
    elif extra_y_test_cols is not None:
        X_test, y_test = data.drop(extra_y_test_cols, axis=1), data[extra_y_test_cols]
    
    if task == constants.SupportedTask.NER:
        if len(X_test.columns) > 1 and "tokens" not in X_test.columns:
            message = "Too many feature columns in dataset. Only 1 feature column should be passed for NER."
            raise ValueError(message)
        if len(X_test.columns) > 1:
            X_test = X_test["tokens"]
        if len(X_test.columns) == 1:
            if isinstance(X_test[X_test.columns[0]].iloc[0], list):
                X_test[X_test.columns[0]] = X_test[X_test.columns[0]].apply(lambda x: " ".join(x))
            if isinstance(X_test[X_test.columns[0]].iloc[0], np.ndarray):
                X_test[X_test.columns[0]] = X_test[X_test.columns[0]].apply(lambda x: " ".join(x.tolist()))
        if isinstance(X_test, pd.Series):
            X_test = X_test.to_frame()
    if _has_multiple_output and y_test is not None and not isinstance(y_test.iloc[0], str):
        if isinstance(y_test.iloc[0], np.ndarray):
            y_test = y_test.apply(lambda x: x.tolist())
        y_test = y_test.astype(str)

    if task == constants.SupportedTask.QnA and y_test is not None:
        if isinstance(y_test.iloc[0], dict):
            # Extracting only the first one for now
            # TODO: Fix this post PrP
            y_test = y_test.apply(lambda x: x["text"][0] if len(x["text"]) > 0 else "")
        elif isinstance(y_test.iloc[0], list) or isinstance(y_test.iloc[0], np.ndarray):
            y_test = y_test.apply(lambda x: x[0])
        if not isinstance(y_test.iloc[0], str):
            message = "Ground Truths for Question-answering should be a string or an array. " \
                      "Found: " + type(y_test.iloc[0])
            raise ValueError(message)
    if task == constants.SupportedTask.FILL_MASK and y_test is not None:
        if isinstance(y_test.iloc[0], np.ndarray) or isinstance(y_test.iloc[0], list):
            y_test = y_test.apply(lambda x: tuple(x))
        if not isinstance(y_test.iloc[0], str) and not isinstance(y_test.iloc[0], tuple):
            message = "Ground Truths for Fill-Mask should be a string or an array found " + type(y_test.iloc[0])
            raise ValueError(message)

    if y_test is not None:
        y_test = y_test.values

    return X_test, y_test

def parse_input_ground_truth_col(col_name):
    """Parse input ground truth columns."""
    extra_cols = None
    if col_name and len(col_name) == 0:
        col_name = None
    if col_name is not None:
        col_name, extra_cols = col_name[0].strip(), col_name[1:]
        # Adding this to be consistent with how it is being used elsewhere, ideally it should be ""
        col_name = None if col_name == "" else col_name

        extra_cols = [i.strip() for i in extra_cols if i and not i.isspace()]
        extra_cols = None if len(extra_cols) == 0 else extra_cols
    return col_name, extra_cols