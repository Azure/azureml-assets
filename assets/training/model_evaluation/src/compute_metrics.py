# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main script for compute metrics."""

import azureml.evaluate.mlflow as aml_mlflow
import os
import glob
import pandas as pd
import numpy as np
from azureml.telemetry.activity import log_activity

import constants
from exceptions import (
    DataLoaderException,
    DataValidationException,
    ComputeMetricsException,
    swallow_all_exceptions,
)
from error_definitions import (
    ComputeMetricsInternalError,
    BadForecastData,
    InvalidGroundTruthColumnName,
    InvalidGroundTruthColumnNameData,
    InvalidPredictionColumnNameData,
    BadInputData,
)
from logging_utilities import custom_dimensions, current_run, get_logger, log_traceback
from utils import (
    ArgumentParser,
    fetch_compute_metrics_args,
    check_and_return_if_mltable,
    read_data,
    evaluate_predictions,
    parse_input_ground_truth_col
)
from validation import validate_compute_metrics_args
from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
from azureml._common._error_definition.azureml_error import AzureMLError

logger = get_logger(name=__name__)
custom_dimensions.app_name = constants.TelemetryConstants.COMPUTE_METRICS_NAME
# current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
custom_dims_dict = vars(custom_dimensions)


class ComputeMetricsRunner:
    """Model Evaluation Runner."""

    def __init__(self,
                 task: str,
                 ground_truth: str,
                 predictions: str,
                 prediction_probabilities: str,
                 output: str,
                 config: dict = None,
                 is_ground_truth_mltable: str = None,
                 is_predictions_mltable: str = None,
                 is_prediction_probabilities_mltable: str = None,
                 ground_truths_column_name: str = None,
                 predictions_column_name: str = constants.PREDICTIONS_COLUMN_NAME):
        """__init__.

        Args:
            task (str): _description_
        """
        self.task = task
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.predictions_probabilities = prediction_probabilities if prediction_probabilities != '' else None
        self.output = output
        self.is_ground_truth_mltable = is_ground_truth_mltable
        self.is_predictions_mltable = is_predictions_mltable
        self.is_predictions_probabilities_mltable = is_prediction_probabilities_mltable
        if self.task != constants.TASK.CHAT_COMPLETION and ground_truths_column_name is not None:
            self.ground_truths_column_name, self.extra_y_test_cols = \
                parse_input_ground_truth_col(ground_truths_column_name)
        else:
            self.ground_truths_column_name = None
            self.extra_y_test_cols = None
        self.predictions_column_name = predictions_column_name

        self.config = config
        self._is_multilabel = self.config.get("multilabel", False)
        self._has_multiple_output = self._is_multilabel or self.task == constants.TASK.NER

    def read_multiple_files(self, path):
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
            df = read_data(file_path=file_path, is_mltable=False)
            df = list(df)[0]
            dfs.append(df)
        if not dfs:
            exception = DataLoaderException._with_error(
                AzureMLError.create(BadInputData, error="No JSON Lines files found in folder.")
            )
            log_traceback(exception, logger)
            raise exception
        data = pd.concat(dfs, ignore_index=True)
        return iter([data])

    def load_data(self):
        """Load Test data.

        Returns:
            _type_: _description_
        """
        if os.path.isdir(self.ground_truth) and not self.is_ground_truth_mltable:
            ground_truth = self.read_multiple_files(self.ground_truth)
        else:
            ground_truth = read_data(self.ground_truth, is_mltable=self.is_ground_truth_mltable)
        ground_truth = list(ground_truth)[0]

        ground_truth = filter_ground_truths(ground_truth, self.task, self.ground_truths_column_name,
                                            self.config)

        if os.path.isdir(self.predictions) and not self.is_predictions_mltable:
            predictions = self.read_multiple_files(path=self.predictions)
        else:
            predictions = read_data(self.predictions, is_mltable=self.is_predictions_mltable)
        predictions = list(predictions)[0]
        predictions = filter_predictions(predictions, self.task, self.predictions_column_name)

        predictions_probabilities = None
        if self.predictions_probabilities is not None:
            if os.path.isdir(self.predictions_probabilities) and not self.is_predictions_probabilities_mltable:
                predictions_probabilities = self.read_multiple_files(path=self.predictions_probabilities)
            else:
                predictions_probabilities = read_data(self.predictions_probabilities,
                                                      is_mltable=self.is_predictions_probabilities_mltable)
            predictions_probabilities = list(predictions_probabilities)[0]
        self.ground_truth, self.predictions, self.predictions_probabilities = \
            ground_truth, predictions, predictions_probabilities

    def compute_metrics(self):
        """Compute Metrics Mode."""
        try:
            ground_true_regressors = None
            self.rename_columns = {}
            if self.task == constants.TASK.FORECASTING:
                ground_truth = self.ground_truth.pop(self.ground_truths_column_name).values
                ground_true_regressors = self.ground_truth
                self.ground_truth = ground_truth
                forecast_origin_column = self.config.get(
                    constants.ForecastingConfigContract.FORECAST_ORIGIN_COLUMN_NAME,
                    constants.ForecastColumns._FORECAST_ORIGIN_COLUMN_DEFAULT)
                if isinstance(self.predictions, pd.DataFrame):
                    # In rolling forecast scenarios we will need to convert the horizon origins to datetime
                    # and give it a default name. Later we will rename this column back.
                    if forecast_origin_column in ground_true_regressors:
                        ground_true_regressors[forecast_origin_column] = pd.to_datetime(
                            ground_true_regressors[forecast_origin_column], unit='ms')
                        if forecast_origin_column != constants.ForecastColumns._FORECAST_ORIGIN_COLUMN_DEFAULT:
                            ground_true_regressors.rename({
                                forecast_origin_column: constants.ForecastColumns._FORECAST_ORIGIN_COLUMN_DEFAULT},
                                inplace=True, axis=1)
                            self.rename_columns = {
                                constants.ForecastColumns._FORECAST_ORIGIN_COLUMN_DEFAULT: forecast_origin_column}
                    if self.predictions_column_name:
                        self.predictions = self.predictions.pop(self.predictions_column_name)
            extra_args = {}
            if self.extra_y_test_cols is not None:
                extra_args[constants.DataFrameParams.Extra_Cols] = self.extra_y_test_cols
            if self.ground_truths_column_name is not None:
                extra_args[constants.DataFrameParams.Ground_Truth_Column_Name] = self.ground_truths_column_name

            return evaluate_predictions(self.ground_truth, self.predictions, self.predictions_probabilities,
                                        self.task, self.config, ground_true_regressors, **extra_args)
        except Exception as e:
            exception = ComputeMetricsException._with_error(
                AzureMLError.create(ComputeMetricsInternalError, error=repr(e)),
                inner_exception=e
            )
            log_traceback(exception, logger)
            raise exception

    def log_and_write_outputs(self, result):
        """Log and Save Outputs."""
        if result:
            scalar_metrics = result.metrics
            logger.info("Computed metrics:")
            for metrics, value in scalar_metrics.items():
                formatted = f"{metrics}: {value}"
                logger.info(formatted)
            if self.task == constants.TASK.FORECASTING and self.rename_columns \
                    and 'forecast_time_series_id_distribution_table' in result.artifacts:
                artifact_content = result.artifacts['forecast_time_series_id_distribution_table'].content
                ts_id_table = pd.DataFrame(artifact_content['data'])
                ts_id_table.rename(self.rename_columns, axis=1, inplace=True)
                artifact_content['data'] = ts_id_table.to_dict(orient='recrds')
                new_table = JsonEvaluationArtifact(
                    uri=result.artifacts['forecast_time_series_id_distribution_table'].uri,
                    content=artifact_content)
                result.artifacts['forecast_time_series_id_distribution_table'] = new_table
            result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))


def filter_ground_truths(data, task_type, column_name=None, config=None):
    """Read Json file utility function.

    Args:
        data (_type_): _description_
        column_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if task_type in [constants.TASK.IMAGE_INSTANCE_SEGMENTATION, constants.TASK.IMAGE_OBJECT_DETECTION] or \
            (task_type == constants.TASK.TEXT_GENERATION and
             config.get(constants.TextGenerationColumns.SUBTASKKEY, "") == constants.SubTask.CODEGENERATION):
        # do not filter as these contains multiple required columns
        return data
    #  for Question-Answering checking for multiple columns in ground truth
    if task_type == constants.TASK.QnA and column_name:
        if isinstance(data[data.columns[0]][0], dict) and len(data[data.columns[0]][0].keys()) > 1:
            try:
                if isinstance(data, pd.DataFrame):
                    logger.warning("Multiple ground truths are not supported for the \
                                   Question and Answering currently.\
                                   Considering only the first ground truth in case of multiple values.")
                    data[data.columns[0]] = data[data.columns[0]].apply(
                        lambda x: x[column_name][0] if len(x[column_name]) > 0 else ""
                    )
            except Exception as e:
                exception = DataValidationException._with_error(
                    AzureMLError.create(InvalidGroundTruthColumnNameData),
                    inner_exception=e
                )
                log_traceback(exception, logger)
                raise exception
        if column_name in data.columns:
            if isinstance(data[column_name].iloc[0], list) or isinstance(data[column_name].iloc[0], np.ndarray):
                logger.warning("Multiple ground truths are not supported for the Question and Answering currently.\
                                Considering only the first ground truth in case of multiple values.")
                data[column_name] = data[column_name].apply(lambda x: x[0])
    if len(data.columns) > 1:
        if column_name:
            if column_name in data.columns:
                data = data[[column_name]]
            else:
                exception = DataValidationException._with_error(
                    AzureMLError.create(InvalidGroundTruthColumnNameData)
                )
                log_traceback(exception, logger)
                raise exception
        else:
            exception = DataValidationException._with_error(
                AzureMLError.create(InvalidGroundTruthColumnName)
            )
            log_traceback(exception, logger)
            raise exception

    return data


def filter_predictions(data, task_type, column_name):
    """Read Json file utility function.

    Args:
        data (_type_): _description_
        column_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    #  for Question-Answering checking for multiple columns in ground truth
    if task_type == constants.TASK.QnA:
        if isinstance(data[data.columns[0]][0], dict) and len(data[data.columns[0]][0].keys()) > 1:
            try:
                if isinstance(data, pd.DataFrame):
                    logger.warning("Multiple predictions are not supported for the \
                                   Question and Answering currently.\
                                   Considering only the first prediction in case of multiple values.")
                    data[data.columns[0]] = data[data.columns[0]].apply(
                        lambda x: x[column_name][0] if len(x[column_name]) > 0 else ""
                    )
            except Exception as e:
                exception = DataValidationException._with_error(
                    AzureMLError.create(InvalidPredictionColumnNameData),
                    inner_exception=e
                )
                log_traceback(exception, logger)
                raise exception
        if column_name in data.columns:
            if isinstance(data[column_name].iloc[0], list) or isinstance(data[column_name].iloc[0], np.ndarray):
                logger.warning("Multiple predictions are not supported for the Question and Answering currently.\
                                Considering only the first prediction in case of multiple values.")
                data[column_name] = data[column_name].apply(lambda x: x[0])
    if len(data.columns) > 1:
        logger.info("Multiple columns found. Picking only prediction column.")
        if column_name in data.columns:
            data = data[[column_name]]
        else:
            exception = DataValidationException._with_error(
                AzureMLError.create(InvalidPredictionColumnNameData)
            )
            log_traceback(exception, logger)
            raise exception

    return data


@swallow_all_exceptions(logger)
def run():
    """Entry point for compute metrics component."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS)
    parser.add_argument("--ground_truths", type=str, dest="ground_truths", required=True)
    parser.add_argument("--ground_truths_column_name", type=str,
                        dest="ground_truths_column_name", required=False, default=None)
    parser.add_argument("--predictions", type=str, dest="predictions", required=True)
    parser.add_argument("--predictions_column_name", type=str,
                        dest="predictions_column_name", required=False, default=None)
    parser.add_argument("--prediction_probabilities", type=str, dest="prediction_probabilities",
                        required=False, default="")
    parser.add_argument("--config_str", type=str, dest="config_str", required=False, default=None)
    parser.add_argument("--config-file-name", type=str, dest="config_file_name", required=False, default="")

    # Outputs
    parser.add_argument("--output", type=str, dest="output")

    args, unknown_args_ = parser.parse_known_args()

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args.__dict__))
        validate_compute_metrics_args(args)
        config = fetch_compute_metrics_args(args.config, args.task)

        ground_truths = args.ground_truths
        is_ground_truths_mltable = check_and_return_if_mltable(ground_truths)
        predictions = args.predictions
        is_predictions_mltable = check_and_return_if_mltable(predictions)
        prediction_probabilities = args.prediction_probabilities
        is_prediction_probabilities_mltable = check_and_return_if_mltable(
            args.prediction_probabilities
        )

        if args.task == constants.TASK.FORECASTING:
            if not args.ground_truths_column_name:
                # If the ground true column name was not provided, we will try to take it from the config.
                args.ground_truths_column_name = config.get('ground_truths_column_name', '')
            if not args.predictions_column_name:
                args.predictions_column_name = config.get('predictions_column_name', '')
            if not args.ground_truths_column_name or (not is_ground_truths_mltable and not args.ground_truths):
                exception = DataValidationException._with_error(
                    AzureMLError.create(BadForecastData)
                )
                log_traceback(exception, logger)
                raise exception

    with log_activity(logger, constants.TelemetryConstants.INITIALISING_RUNNER,
                      custom_dimensions=custom_dims_dict):
        runner = ComputeMetricsRunner(
            task=args.task,
            ground_truth=ground_truths,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            output=args.output,
            config=config,
            is_ground_truth_mltable=is_ground_truths_mltable,
            is_predictions_mltable=is_predictions_mltable,
            is_prediction_probabilities_mltable=is_prediction_probabilities_mltable,
            ground_truths_column_name=args.ground_truths_column_name,
            predictions_column_name=args.predictions_column_name
        )

    with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        try:
            logger.info("Loading Data.")
            runner.load_data()
        except Exception as e:
            exception = DataLoaderException._with_error(
                AzureMLError.create(BadInputData, error=repr(e)),
                inner_exception=e
            )
            log_traceback(exception, logger)
            raise exception

    with log_activity(logger, activity_name=constants.TelemetryConstants.COMPUTE_METRICS_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Computing metrics.")
        result = runner.compute_metrics()

    logger.info("Logging and Saving outputs.")
    runner.log_and_write_outputs(result)

    test_run.add_properties(properties=constants.RUN_PROPERTIES)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ == "__main__":
    run()
