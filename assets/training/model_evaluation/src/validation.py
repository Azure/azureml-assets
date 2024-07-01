# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for argument validation."""
from exceptions import (
    ModelValidationException,
    DataValidationException,
    ArgumentValidationException,
)
from constants import ALL_TASKS, TASK, ArgumentLiterals
from image_constants import SettingLiterals
from logging_utilities import get_logger, log_traceback, get_azureml_exception
from utils import assert_and_raise, read_config, read_config_str, get_sample_data_and_column_names
from error_definitions import (
    InvalidTaskType,
    InvalidModel,
    BadInputColumnData,
    BadLabelColumnData,
    BadFeatureColumnData,
    InvalidData,
    InvalidFileInputSource,
    BadEvaluationConfig,
    InvalidGroundTruthColumnNameCodeGen,
)
from azureml.evaluate.mlflow.models import Model
from azureml.metrics._score import validate_y_test
import constants

logger = get_logger(name=__name__)


def _validate_model(args):
    """Validate Model.

    Args:
        args (_type_): _description_
    """
    logger.info("Validating Model is passed correctly.")
    try:
        _ = Model.load(args[ArgumentLiterals.MLFLOW_MODEL])
    except Exception as e:
        exception = get_azureml_exception(ModelValidationException, InvalidModel, e)
        log_traceback(exception, logger)
        raise exception


def _validate_task(args):
    """Validate Task selected.

    Args:
        args (_type_): _description_
    """
    logger.info("Validating Task Type: " + args[ArgumentLiterals.TASK])
    message_kwargs = {"TaskName": args[ArgumentLiterals.TASK]}
    assert_and_raise(
        condition=args[ArgumentLiterals.TASK] in ALL_TASKS,
        exception_cls=ArgumentValidationException,
        error_cls=InvalidTaskType,
        **message_kwargs
    )


def _validate_data_passed(data, input_port):
    message_kwargs = {"input_port": input_port}
    assert_and_raise(
        condition=(data is not None and data != ""),
        exception_cls=DataValidationException,
        error_cls=InvalidData,
        **message_kwargs
    )


def _validate_input_source_mount(data, input_port):
    message_kwargs = {"input_port": input_port}
    assert_and_raise(
        condition=(not data.startswith("azureml://")),
        exception_cls=DataValidationException,
        error_cls=InvalidFileInputSource,
        **message_kwargs
    )


def _validate_test_data(args):
    logger.info("Validating Test Data is passed with correct mount.")
    data = args[ArgumentLiterals.DATA]
    _validate_data_passed(data, "test_data")
    _validate_input_source_mount(data, "test_data")


def _validate_compute_metrics_data(args):
    logger.info("Validating Predictions is passed with correct mount.")
    predictions = args[ArgumentLiterals.PREDICTIONS]
    _validate_data_passed(predictions, "predictions")
    _validate_input_source_mount(predictions, "predictions")

    if args[ArgumentLiterals.GROUND_TRUTHS] is not None:
        logger.info("Validating Ground Truths is passed with correct mount.")
        ground_truths = args[ArgumentLiterals.GROUND_TRUTHS]
        _validate_data_passed(ground_truths, "ground_truths")
        _validate_input_source_mount(ground_truths, "ground_truths")


def _validate_config(args):
    logger.info("Reading and Validating Config.")
    if args[ArgumentLiterals.CONFIG_FILE_NAME]:
        _validate_input_source_mount(args[ArgumentLiterals.CONFIG_FILE_NAME], "evaluation_config")
    file_config = read_config(args[ArgumentLiterals.CONFIG_FILE_NAME])
    str_config = read_config_str(args[ArgumentLiterals.CONFIG_STR])

    try:
        config = dict(file_config)
        str_config = dict(str_config)
        if args[ArgumentLiterals.CONFIG_FILE_NAME] and args[ArgumentLiterals.CONFIG_STR]:
            logger.warning("Both evaluation_config and evaluation_config_params are passed. \
                           Overriding evaluation_config file params with evaluation_config string params.")
        config.update(str_config)
        args[ArgumentLiterals.CONFIG] = config
    except Exception as e:
        message = "Unable to load Evaluation Config. Config passed is not JSON serialized."
        exception = get_azureml_exception(DataValidationException, BadEvaluationConfig, e, error=repr(e))
        log_traceback(exception, logger, message)
        raise exception


def _validate_batch_size(args):
    """Validate batch size.

    Args:
        args (_type_): _description_
    """
    if ArgumentLiterals.BATCH_SIZE in args:
        logger.info("Validating batch_size.")
        if args[ArgumentLiterals.BATCH_SIZE] is not None and args[ArgumentLiterals.BATCH_SIZE] < 1:
            logger.warning("Batch size should be > 0. Ignoring parameter.")
            args[ArgumentLiterals.BATCH_SIZE] = None


def validate_common_args(args):
    """Validate Compute Metrics Args.

    Args:
        args (_type_): _description_
    """
    _validate_task(args)
    _validate_config(args)


def validate_model_prediction_args(args):
    """Validate args for model prediction.

    Args:
        args (_type_): _description_
    """
    _validate_model(args)
    _validate_test_data(args)
    _validate_batch_size(args)


def validate_compute_metrics_args(args):
    """Validate Compute Metrics Args.

    Args:
        args (_type_): _description_
    """
    _validate_compute_metrics_data(args)


def validate_compute_metrics_label_column_arg(args):
    """Validate Label Column for Compute Metrics.

    Args:
        args (_type_): _description_
    """
    label_column = args[ArgumentLiterals.LABEL_COLUMN_NAME]
    if label_column is None or len(label_column) == 0:
        try:
            validate_y_test(args[ArgumentLiterals.TASK], None, args[ArgumentLiterals.CONFIG])
        except Exception as e:
            exception = get_azureml_exception(DataValidationException, BadLabelColumnData, e)
            log_traceback(exception, logger)
            raise exception


def validate_input_column_names(input_column_names, data):
    """Validate input Column.

    Args:
        input_column_names (_type_): _description_
        data (_type_): _description_
    """
    logger.info("Validating input columns in data.")
    if len(input_column_names) == 0:
        exception = get_azureml_exception(DataValidationException, BadInputColumnData, None)
        log_traceback(exception, logger)
        raise exception
    try:
        data = data[input_column_names]
    except Exception as e:
        message_kwargs = {
            "column": "input_columns", "keep_columns": str(input_column_names), "data_columns": str(list(data.columns))
        }
        exception = get_azureml_exception(DataValidationException, BadFeatureColumnData, e, **message_kwargs)
        log_traceback(exception, logger)
        raise exception


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
        message_kwargs = {
            "column": "label_column", "keep_columns": str(label_column_names), "data_columns": str(list(data.columns))
        }
        exception = get_azureml_exception(DataValidationException, BadFeatureColumnData, e, **message_kwargs)
        log_traceback(exception, logger)
        raise exception


def validate_and_get_columns(args):
    """Validate and return column names.

    Args:
        args (_type_): _description_
    """
    logger.info("Reading top row in data for column name extraction and validation.")
    data, input_column_names, label_column_name, extra_y_test_cols = get_sample_data_and_column_names(args)

    task = args[ArgumentLiterals.TASK]
    config = args[ArgumentLiterals.CONFIG]

    if task in constants.IMAGE_TASKS:
        # Vision datasets must have an image_url and a label column. The input columns for model prediction will be
        # constructed from these two (pass through in most of the cases).
        validate_input_column_names([SettingLiterals.IMAGE_URL, SettingLiterals.LABEL], data)
    else:
        validate_input_column_names(input_column_names, data)

    if task == TASK.TEXT_GENERATION:
        if config.get(constants.TextGenerationColumns.SUBTASKKEY, "") == constants.SubTask.CODEGENERATION:
            # Ensure that user always has "," in label_col_name
            if extra_y_test_cols is None and label_column_name is None:
                exception = get_azureml_exception(DataValidationException, InvalidGroundTruthColumnNameCodeGen, None)
                log_traceback(exception, logger)
                raise exception

    extra_cols_tasks = [TASK.TEXT_GENERATION, TASK.QnA, TASK.CHAT_COMPLETION]
    if extra_y_test_cols is not None and task not in extra_cols_tasks:
        logger.info(f"extra_y_test_cols not supported for task type {task}. Setting to None.")
        extra_y_test_cols = None

    cols = []
    if label_column_name is not None:
        cols += [label_column_name]
    if extra_y_test_cols is not None:
        cols += extra_y_test_cols
    validate_label_column_names(cols, data)

    return input_column_names, label_column_name, extra_y_test_cols


def validate_Xy(X_test, y_test):
    """Validate Input X and Y.

    Args:
        X_test (_type_): _description_
        y_test (_type_): _description_
    """
    # _validate_data_passed(X_test, "test_data")
    message_kwargs = {"input_port": "test_data"}
    assert_and_raise(
        condition=(X_test is not None and len(X_test) != 0),
        exception_cls=DataValidationException,
        error_cls=InvalidData,
        **message_kwargs
    )
    assert_and_raise(
        condition=y_test is not None,
        exception_cls=DataValidationException,
        error_cls=BadLabelColumnData
    )
