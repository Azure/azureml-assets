# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for argument validation."""
from exceptions import (
    ModelValidationException,
    DataValidationException,
    ArgumentValidationException,
    AzureMLException,
)
from constants import ALL_TASKS, TASK
from logging_utilities import get_logger, log_traceback
from utils import assert_and_raise, read_config, read_config_str
from typing import Union
from error_definitions import (
    InvalidTaskType,
    InvalidGroundTruthData,
    InvalidPredictionsData,
    InvalidModel,
    BadInputData,
    BadLabelColumnData,
    BadFeatureColumnData,
    EmptyInputData,
    InvalidTestData,
    InvalidFileInputSource,
    BadEvaluationConfig,
)
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.evaluate.mlflow.models import Model

logger = get_logger(name=__name__)


def _validate_model(args):
    """Validate Model.

    Args:
        args (_type_): _description_
    """
    logger.info("Validating Model is passed")

    try:
        _ = Model.load(args.mlflow_model)
    except Exception as e:
        exception = ModelValidationException._with_error(
            AzureMLError.create(InvalidModel, error=repr(e)),
            inner_exception=e
        )
        log_traceback(exception, logger)
        raise exception


def _validate_task(args):
    """Validate Task selected.

    Args:
        args (_type_): _description_
    """
    logger.info("Validating Task Type: " + args.task)
    assert_and_raise(
        condition=args.task in ALL_TASKS,
        exception_cls=ArgumentValidationException,
        error_cls=InvalidTaskType,
        message_kwargs={"TaskName": args.task}
    )


def _validate_input_source_mount(data, input_port):
    assert_and_raise(
        condition=(not data.startswith("azureml://")),
        exception_cls=DataValidationException,
        error_cls=InvalidFileInputSource,
        message_kwargs={"input_port": input_port}
    )


def _validate_test_data(args):
    logger.info("Validating Test Data is passed")
    _data = args.data
    assert_and_raise(
        condition=(_data is not None and _data != ""),
        exception_cls=DataValidationException,
        error_cls=InvalidTestData
    )
    _validate_input_source_mount(_data, "test_data")


def _validate_compute_metrics_data(args):
    # _, predictions = check_and_return_if_mltable(args.predictions, args.predictions_mltable)
    predictions = args.predictions
    assert_and_raise(
        condition=(predictions is not None and predictions != ""),
        exception_cls=DataValidationException,
        error_cls=InvalidPredictionsData
    )
    _validate_input_source_mount(predictions, "predictions")

    if args.task != TASK.FILL_MASK:
        # _, ground_truths = check_and_return_if_mltable(args.ground_truths, args.ground_truths_mltable)
        ground_truths = args.ground_truths
        assert_and_raise(
            condition=(ground_truths is not None and ground_truths != ""),
            exception_cls=DataValidationException,
            error_cls=InvalidGroundTruthData
        )
        _validate_input_source_mount(ground_truths, "ground_truths")


def _validate_config(args):
    if args.config_file_name:
        _validate_input_source_mount(args.config_file_name, "evaluation_config")
    file_config = read_config(args.config_file_name)
    str_config = read_config_str(args.config_str)

    try:
        config = dict(file_config)
        str_config = dict(str_config)
        if args.config_file_name and args.config_str:
            logger.warning("Both evaluation_config and evaluation_config_params are passed. \
                           Overriding evaluation_config file params with evaluation_config string params.")
        config.update(str_config)
        args.config = config
    except Exception as e:
        message = "Unable to load Evaluation Config. Config passed is not JSON serialized."
        exception = DataValidationException._with_error(
            AzureMLError.create(BadEvaluationConfig, error=repr(e))
        )
        log_traceback(exception=exception, logger=logger, message=message)
        raise exception


def _validate_batch_size(args):
    if "batch_size" in args:
        if args.batch_size is not None and args.batch_size < 1:
            logger.warning("Batch size should be > 0. Ignoring parameter.")
            args.batch_size = None


def validate_args(args):
    """Validate All args.

    Args:
        args (_type_): _description_
    """
    _validate_model(args)
    _validate_task(args)
    _validate_test_data(args)
    _validate_config(args)
    _validate_batch_size(args)


def validate_compute_metrics_args(args):
    """Validate Compute Metrics Args.

    Args:
        args (_type_): _description_
    """
    _validate_task(args)
    _validate_compute_metrics_data(args)
    _validate_config(args)


def validate_Xy(X_test, y_test):
    """Validate Input X and Y.

    Args:
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    # message = "Invalid data. No feature matrix found."
    assert_and_raise(
        condition=X_test is not None,
        exception_cls=DataValidationException,
        error_cls=InvalidTestData
    )
    assert_and_raise(
        condition=y_test is not None,
        exception_cls=DataValidationException,
        error_cls=BadLabelColumnData
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


def _validate(data, input_column_names=None, label_column_name=None, extra_cols=None, batch_size=None):
    try:
        # If input_column_names are not sent as argument we are retaining all columns
        if input_column_names is None:
            input_column_names = list(data.columns)
            if label_column_name in input_column_names:
                input_column_names.remove(label_column_name)
        if extra_cols is not None:
            input_column_names += extra_cols
        if label_column_name is not None:
            input_column_names += [label_column_name]
        data = _clean_and_validate_dataset(data, input_column_names, batch_size)

    except Exception as e:
        if isinstance(e, AzureMLException):
            exception = e
        else:
            error_message = f"Failed to open test data with following error {repr(e)}"
            exception = DataValidationException._with_error(
                AzureMLError.create(BadInputData, error=repr(e)),
                inner_exception=e
            )
            log_traceback(exception, logger, error_message, is_critical=True)
        raise exception
    return data


def _clean_and_validate_dataset(data, keep_columns, batch_size=None):
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
    # message = "input_column_names is not a subset of input test dataset columns.\
    #              input_column_names include " + str(keep_columns) + " whereas data has " + str(list(data.columns))
    assert_and_raise(
        condition=sorted(keep_columns) == sorted(data.columns),
        exception_cls=DataValidationException,
        error_cls=BadFeatureColumnData,
        message_kwargs={"keep_columns": str(keep_columns), "data_columns": str(list(data.columns))}
    )

    # remove the null values
    pre_filter_examples = len(data)
    logger.info("Filtering rows with 'None' values")
    logger.info(f"Number of examples before filter: {pre_filter_examples}")
    # TODO support batched=True and handle processing multiple examples in lambda
    data['to_filter'] = data.apply(lambda x: all(_check_if_non_empty(x[col]) for col in keep_columns), axis=1)
    data = data.loc[data['to_filter']]
    data = data.drop('to_filter', axis=1)
    post_filter_examples = len(data)
    logger.info(f"Number of examples after postprocessing: {post_filter_examples}")
    if post_filter_examples == 0 and batch_size is None:
        exception = DataValidationException._with_error(
            AzureMLError.create(EmptyInputData)
        )
        log_traceback(exception=exception, logger=logger)
        raise exception

    # logging
    if pre_filter_examples == post_filter_examples:
        logger.info("None of the examples are filtered")
    else:
        logger.info(
            f"{pre_filter_examples - post_filter_examples} examples are discarded "
            f"as atleast one of the columns in the data is empty"
        )

    return data
