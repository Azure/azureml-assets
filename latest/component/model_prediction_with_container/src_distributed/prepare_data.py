# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# flake8: noqa: E702

"""Prepare data."""
import sys

sys.path.append("/src/")

from data_utils import read_model_prediction_data, parse_input_ground_truth_col, prepare_data
from argparse import ArgumentParser
from logging_utilities import get_logger
from llm.optimized.inference.constants import ALL_TASKS
from local_constants import ArgumentLiterals
from itertools import repeat
from typing import Union

import pandas as pd
import os

logger = get_logger(name=__name__)


def get_column_names(args, data):
    """Get Column names from test data."""
    task = args[ArgumentLiterals.TASK]
    # If input_column_names are not sent as argument we are retaining all columns
    label_column_name = args[ArgumentLiterals.LABEL_COLUMN_NAME]
    label_column_name, extra_y_test_cols = parse_input_ground_truth_col(label_column_name)

    input_column_names = args[ArgumentLiterals.INPUT_COLUMN_NAMES]
    if input_column_names is None or len(input_column_names) == 0:
        input_column_names = list(data.columns)
        if label_column_name is not None and label_column_name in input_column_names:
            input_column_names.remove(label_column_name)
        if extra_y_test_cols is not None:
            for col in extra_y_test_cols:
                if col in input_column_names:
                    input_column_names.remove(col)
    return input_column_names, label_column_name, extra_y_test_cols


def validate_input_column_names(input_column_names, data):
    """Validate input Column.

    Args:
        input_column_names (_type_): _description_
        data (_type_): _description_
    """
    logger.info("Validating input columns in data.")
    if len(input_column_names) == 0:
        raise ValueError("No Input Column names found in data.")
    try:
        data = data[input_column_names]
    except Exception as e:
        raise e


def validate_label_column_names(label_column_names, data):
    """Validate Label Column.

    Args:
        label_column_names (_type_): _description_
        data (_type_): _description_
    """
    try:
        logger.info("Validating label columns in data.")
        data = data[label_column_names]
    except Exception as e:
        raise e

def validate_and_get_columns(args):
    """Validate and return column names.

    Args:
        args (_type_): _description_
    """
    logger.info("Reading top row in data for validation.")
    data = list(read_model_prediction_data(args["data"], nrows=1))[0]
    input_column_names, label_column_name, extra_y_test_cols = get_column_names(args, data)

    validate_input_column_names(input_column_names, data)

    
    cols = []
    if label_column_name is not None:
        cols += [label_column_name]
    if extra_y_test_cols is not None:
        cols += extra_y_test_cols
    validate_label_column_names(cols, data)

    return input_column_names, label_column_name, extra_y_test_cols


def _check_if_non_empty(val: Union[str, list, int]) -> bool:
    # For the supported tasks val will be the following
    # Single Label - int, str
    # Multi Label - int, str
    # Regression - str, float
    # NER, POS, Chunking - list
    # Summarization, Translation - str
    # QnA - data validation is `skipped`
    if val is None:
        return False
    if isinstance(val, (str, list)):
        return len(val) != 0

    return True


def _clean_and_validate_dataset(data, keep_columns, batch_size=None):
    """
    Clean the data for irrelevant columns and null values.

    Args:
        data: Incoming Data
        keep_columns: Columns to extract from data

    Returns: Data

    """
    try:
        logger.info("Filtering data columns from input columns.")
        data = data[keep_columns]
    except Exception as e:
        raise e

    # remove the null values
    pre_filter_examples = len(data)
    if pre_filter_examples == 0:
        raise ValueError("No examples to filter.")

    logger.info("Filtering rows with 'None' values")
    logger.info(f"Number of examples before filter: {pre_filter_examples}")

    # TODO support batched=True and handle processing multiple examples in lambda
    try:
        data['to_filter'] = data.apply(lambda x: all(_check_if_non_empty(x[col]) for col in keep_columns), axis=1)
        data = data.loc[data['to_filter']]
        data = data.drop('to_filter', axis=1)
        post_filter_examples = len(data)
    except Exception as e:
        raise e

    logger.info(f"Number of examples after postprocessing: {post_filter_examples}")

    # logging
    if pre_filter_examples == post_filter_examples:
        logger.info("None of the examples are filtered")
    else:
        logger.info(
            f"{pre_filter_examples - post_filter_examples} examples are discarded "
            f"as atleast one of the columns in the data is empty"
        )
    if post_filter_examples == 0 and batch_size is None:
        message = "Failed to prepare data with error: No examples left after filtering."
        raise ValueError(message)

    return data


def prepare_prs_data():
    """Entry function of model prediction script."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest=ArgumentLiterals.TASK, required=True, choices=ALL_TASKS)
    parser.add_argument("--data", type=str, dest=ArgumentLiterals.DATA, required=True)
    parser.add_argument("--label-column-name", type=lambda x: x.split(","),
                        dest=ArgumentLiterals.LABEL_COLUMN_NAME, required=False, default=None)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest=ArgumentLiterals.INPUT_COLUMN_NAMES, required=False, default=None)
    parser.add_argument("--config-file-name", type=str, dest=ArgumentLiterals.CONFIG_FILE_NAME,
                        required=False, default=None)
    parser.add_argument("--config_str", type=str, dest=ArgumentLiterals.CONFIG_STR, required=False, default=None)

    parser.add_argument("--batch-size", type=int, dest=ArgumentLiterals.BATCH_SIZE, required=False, default=None)

    # Outputs
    parser.add_argument("--prs-data", type=str, dest=ArgumentLiterals.PRS_DATA)
    parser.add_argument("--ground-truth", type=str, dest=ArgumentLiterals.GROUND_TRUTHS, required=False, default=None)
    args, _ = parser.parse_known_args()
    args = vars(args)

    out_path = args[ArgumentLiterals.PRS_DATA]
    ground_truths_path = args[ArgumentLiterals.GROUND_TRUTHS]
    input_column_names, label_column_name, extra_y_test_cols = validate_and_get_columns(args)
    all_cols = list(input_column_names)
    if label_column_name is not None:
        all_cols += [label_column_name]
    if extra_y_test_cols is not None:
        all_cols += extra_y_test_cols

    data = read_model_prediction_data(file_path=args[ArgumentLiterals.DATA], batch_size=args[ArgumentLiterals.BATCH_SIZE])
    data = map(_clean_and_validate_dataset, data, repeat(all_cols), repeat(args[ArgumentLiterals.BATCH_SIZE]))
    data = map(prepare_data, data, repeat(args[ArgumentLiterals.TASK]), repeat(label_column_name),
                repeat(False), repeat(extra_y_test_cols))

    y_test = pd.DataFrame()
    for idx, (X_test, y_test_batch) in enumerate(data):
        if len(X_test) == 0:
            logger.info("No samples in batch. Skipping.")
            continue
        out_file_path = os.path.join(out_path, f"Input_file_{idx}.parquet")
        X_test.to_parquet(out_file_path)
        logger.info(f"Writing files to {out_file_path}")
        y_test_batch = pd.DataFrame(y_test_batch, columns=[label_column_name])
        y_test = pd.concat([y_test, y_test_batch], axis=0)

    logger.info("Logging Ground Truths")
    y_test.to_json(ground_truths_path, orient="records", lines=True)