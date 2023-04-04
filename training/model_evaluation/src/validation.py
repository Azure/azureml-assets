# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for argument validation."""
import os
import traceback
from exceptions import ModelValidationException, DataValidationException, ArgumentValidationException
from constants import ALL_TASKS, TASK
from logging_utilities import get_logger, log_traceback
from utils import assert_and_raise, check_and_return_if_mltable
from typing import Union


logger = get_logger(name=__name__)


def check_model_uri(model_uri):
    """Validate Model URI.

    Args:
        model_uri (_type_): _description_

    Returns:
        _type_: _description_
    """
    if model_uri.startswith("runs:/"):
        return True
    if model_uri.startswith("models:/"):
        return True
    if os.path.exists(model_uri):
        return True
    return False


def _validate_model(args):
    """Validate Model.

    Args:
        args (_type_): _description_
    """
    assert_and_raise(condition=(len(args.model_uri) > 0) or (args.mlflow_model is not None),
                     exception_cls=ModelValidationException,
                     message="Either Model URI or Mlflow Model is required")

    mlflow_model, model_uri = False, False

    if args.mlflow_model:
        mlflow_model = "MLmodel" in os.listdir(args.mlflow_model)
        if not mlflow_model:
            logger.warn("Invalid mlflow model passed. Trying model_uri.")
            args.mlflow_model = None

    if args.model_uri:
        model_uri = check_model_uri(args.model_uri)
        if not model_uri:
            logger.warn("Invalid model uri passed")
            args.model_uri = ""

    error_message = "Invalid mlflow_model/model_uri passed. See documentation for mlflow_model and model_uri format."
    assert_and_raise(
        condition=(len(args.model_uri) > 0) or (args.mlflow_model is not None),
        exception_cls=ModelValidationException,
        message=error_message
    )


def _validate_task(args):
    """Validate Task selected.

    Args:
        args (_type_): _description_
    """
    # if args.task is None:
    #     return
    assert_and_raise(
        condition=args.task in ALL_TASKS,
        exception_cls=ArgumentValidationException,
        message="Invalid task type. It should be either classification, regression, forecasting or NER"
    )


def _validate_mode(args):
    """Validate Mode.

    Args:
        args (_type_): _description_
    """
    assert_and_raise(
        condition=args.mode in ["predict", "compute_metrics", "score"],
        exception_cls=ArgumentValidationException,
        message="Invalid mode type. It should be either predict, compute_metrics or score"
    )

def _validate_test_data(args):
    _, _data = check_and_return_if_mltable(args.data, args.data_mltable)
    assert_and_raise(
        condition=(_data is not None and _data != ""),
        exception_cls=DataValidationException,
        message="Either test_data or test_data_mltable should be passed."
    )


def _validate_compute_metrics_data(args):
    _, predictions = check_and_return_if_mltable(args.predictions, args.predictions_mltable)
    assert_and_raise(
        condition=(predictions is not None and predictions != ""),
        exception_cls=DataValidationException,
        message="Either predictions or predictions_mltable should be passed."
    )
    if args.task != TASK.FILL_MASK:
        _, ground_truths = check_and_return_if_mltable(args.ground_truths, args.ground_truths_mltable)
        assert_and_raise(
            condition=(ground_truths is not None and ground_truths != ""),
            exception_cls=DataValidationException,
            message="Either ground_truths or ground_truths_mltable should be passed."
        )


def validate_args(args):
    """Validate All args.

    Args:
        args (_type_): _description_
    """
    _validate_model(args)
    _validate_task(args)
    _validate_test_data(args)


def validate_compute_metrics_args(args):
    """Validate Compute Metrics Args.

    Args:
        args (_type_): _description_
    """
    _validate_task(args)
    _validate_compute_metrics_data(args)


def _validate_Xy(X_test, y_test, y_pred, mode):
    """Validate Input X and Y.

    Args:
        X_test (_type_): _description_
        y_test (_type_): _description_
        y_pred (_type_): _description_
        mode (_type_): _description_
    """
    message = "Invalid data. No feature matrix found."
    assert_and_raise(
        condition=X_test is not None,
        exception_cls=DataValidationException,
        message=message
    )
    if mode == "score":
        assert_and_raise(
            condition=y_test is not None,
            exception_cls=DataValidationException,
            message="No label column found in test data. Required for mode 'score'"
        )
    if mode == "compute_metrics":
        assert_and_raise(
            condition=y_pred is not None,
            exception_cls=DataValidationException,
            message="No predictions column name found in test data. Required for mode 'compute_metrics'"
        )


def validate_Xy(X_test, y_test):
    """Validate Input X and Y.

    Args:
        X_test (_type_): _description_
        y_test (_type_): _description_
        y_pred (_type_): _description_
        mode (_type_): _description_
    """
    message = "Invalid data. No feature matrix found."
    assert_and_raise(
        condition=X_test is not None,
        exception_cls=DataValidationException,
        message=message
    )
    assert_and_raise(
        condition=y_test is not None,
        exception_cls=DataValidationException,
        message="No label column found in test data."
    )


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


def _validate(data, input_column_names=None, label_column_name=None):
    try:
        # If input_column_names are not sent as argument we are retaining all columns
        if input_column_names is None:
            input_column_names = list(data.columns)
            if label_column_name in input_column_names:
                input_column_names.remove(label_column_name)
        if label_column_name is not None and label_column_name not in input_column_names:
            data = _clean_and_validate_dataset(data, input_column_names + [label_column_name])
        else:
            data = _clean_and_validate_dataset(data, input_column_names)
    except Exception as e:
        traceback.print_exc()
        error_message = f"Failed to open test data with following error {repr(e)}"
        log_traceback(e, logger, error_message, is_critical=True)
        raise DataValidationException(error_message)
    return data


def _clean_and_validate_dataset(data, keep_columns):
    """
    Clean the data for irrelevant columns and null values.

    Args:
        data: Incoming Data
        keep_columns: Columns to extract from data

    Returns: Data

    """
    data_columns = data.columns
    to_remove_columns = [col.strip() for col in data_columns if col.strip() not in keep_columns]
    data = data.drop(to_remove_columns, axis=1)
    message = "input_column_names is not a subset of input test dataset columns.\
                 input_column_names include "+str(keep_columns)+" whereas data has "+str(list(data.columns))
    assert_and_raise(
        condition=sorted(keep_columns) == sorted(data.columns),
        exception_cls=DataValidationException,
        message=message
    )

    # remove the null values
    pre_filter_examples = data.shape[0]
    logger.info(f"Number of examples before filter: {pre_filter_examples}")
    # TODO support batched=True and handle processing multiple examples in lambda
    data['to_filter'] = data.apply(lambda x: all(_check_if_non_empty(x[col]) for col in keep_columns), axis=1)
    data = data.loc[data['to_filter']]
    data = data.drop('to_filter', axis=1)
    post_filter_examples = data.shape[0]
    logger.info(f"Number of examples after postprocessing: {post_filter_examples}")
    if post_filter_examples == 0:
        raise Exception("Found no examples after data preprocessing")

    # logging
    if pre_filter_examples == post_filter_examples:
        logger.info("None of the examples are filtered")
    else:
        logger.info(
            f"{pre_filter_examples - post_filter_examples} examples are discarded "
            f"as atleast one of the columns in the data is empty"
        )

    return data
