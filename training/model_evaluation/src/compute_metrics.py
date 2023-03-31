# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main script for compute metrics."""

import argparse
import azureml.evaluate.mlflow as aml_mlflow
import os
import glob
import pandas as pd
import numpy as np
from azureml.telemetry.activity import log_activity

import constants
from exceptions import DataLoaderException
from logging_utilities import custom_dimensions, get_logger, log_traceback
from utils import (read_compute_metrics_config,
                   check_and_return_if_mltable,
                   read_data,
                   evaluate_predictions)
from run_utils import TestRun
from validation import validate_compute_metrics_args

logger = get_logger(name=__name__)
current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


class ModelEvaluationRunner:
    """Model Evaluation Runner."""

    def __init__(self,
                 task: str,
                 ground_truth: str,
                 predictions: str,
                 prediction_probabilities: str,
                 output: str,
                 custom_dimensions: dict,
                 config_file: str = None,
                 is_ground_truth_mltable: str = None,
                 is_predictions_mltable: str = None,
                 is_prediction_probabilities_mltable: str = None,
                 ground_truths_column_name: str = None,
                 predictions_column_name: str = None):
        """__init__.

        Args:
            task (str): _description_
            custom_dimensions (dict): _description_
        """
        self.task = task
        self.ground_truth = ground_truth
        self.predictions = predictions
        self.predictions_probabilities = prediction_probabilities if prediction_probabilities != '' else None
        self.config_file = config_file
        self.output = output
        self.is_ground_truth_mltable = is_ground_truth_mltable
        self.is_predictions_mltable = is_predictions_mltable
        self.is_predictions_probabilities_mltable = is_prediction_probabilities_mltable
        self.ground_truths_column_name = ground_truths_column_name
        self.predictions_column_name = predictions_column_name

        self.label_column_name, self.prediction_column_name, self.metrics_config = None, None, {}
        if config_file:
            self.metrics_config = read_compute_metrics_config(config_file, self.task)
        self._is_multilabel = self.metrics_config.get("multilabel", False)
        self.custom_dimensions = custom_dimensions
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
            logger.error("No JSONL files found in data directory")
            raise DataLoaderException(f"No JSONL files found in the directory {path}")
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
        ground_truth = filter_ground_truths(ground_truth, self.task, self.ground_truths_column_name)

        if os.path.isdir(self.predictions) and not self.is_predictions_mltable:
            predictions = self.read_multiple_files(path=self.predictions)
        else:
            predictions = read_data(self.predictions, is_mltable=self.is_predictions_mltable)
        predictions = list(predictions)[0]
        predictions = filter_ground_truths(predictions, self.task, self.predictions_column_name)

        predictions_probabilities = None
        if self.predictions_probabilities is not None:
            if os.path.isdir(self.predictions_probabilities) and not self.is_predictions_probabilities_mltable:
                predictions_probabilities = self.read_multiple_files(path=self.predictions_probabilities)
            else:
                predictions_probabilities = read_data(self.predictions_probabilities,
                                                      is_mltable=self.is_predictions_probabilities_mltable)
            predictions_probabilities = list(predictions_probabilities)[0]
        return ground_truth, predictions, predictions_probabilities

    def fetch_mode_runner(self):
        """Get Runner function.

        Returns:
            _type_: _description_
        """
        return getattr(self)

    def compute_metrics(self):
        """Compute Metrics Mode."""
        with log_activity(logger, constants.TelemetryConstants.COMPUTE_METRICS_NAME,
                          custom_dimensions=self.custom_dimensions):
            result = evaluate_predictions(self.ground_truth, self.predictions, self.predictions_probabilities, 
                                          self.task, self.metrics_config)
            if result:
                scalar_metrics = result.metrics
                logger.info("Computed metrics:")
                for metrics, value in scalar_metrics.items():
                    formatted = f"{metrics}: {value}"
                    logger.info(formatted)
                result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))
        return

    def run(self):
        """Model Evaluation Runner.

        Raises:
            DataLoaderException: _description_
        """
        with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                          custom_dimensions=self.custom_dimensions):
            try:
                print("Loading data ", self.predictions_probabilities, type(self.predictions_probabilities))
                self.ground_truth, self.predictions, self.predictions_probabilities = self.load_data()
            except Exception as e:
                message = "Couldn't load data."
                log_traceback(e, logger, message, True)
                raise DataLoaderException(message, inner_exception=e)

        self.compute_metrics()


def filter_ground_truths(data, task_type, column_name=None):
    """Read Json file utility function.

    Args:
        data (_type_): _description_
        column_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    #  for Question-Answering checking for multiple columns in ground truth
    if task_type == constants.TASK.QnA and column_name:
        if type(data[data.columns[0]][0]) == dict and len(data[data.columns[0]][0].keys()) > 1:
            try:
                if isinstance(data, pd.DataFrame):
                    logger.warning("Multiple ground truths are not supported for the \
                                   Question and Answering currently.\
                                   Considering only the first ground truth in case of multiple values.")
                    data[data.columns[0]] = data[data.columns[0]].apply(
                        lambda x: x[column_name][0] if len(x[column_name]) > 0 else ""
                    )
            except Exception as e:
                message = "Invalid ground truths column name"
                log_traceback(e, logger, message, True)
                raise DataLoaderException(message, inner_exception=e)
        if column_name in data.columns:
            if isinstance(data[column_name].iloc[0], list) or isinstance(data[column_name].iloc[0], np.ndarray):
                logger.warning("Multiple ground truths are not supported for the Question and Answering currently.\
                                Considering only the first ground truth in case of multiple values.")
                data[column_name] = data[column_name].apply(lambda x: x[0])
            if len(data.columns) > 0:
                data = data[[column_name]]

    return data


def test_component():
    """Entry point for compute metrics component."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS)
    parser.add_argument("--ground_truths", type=str, dest="ground_truths", required=False, default="")
    parser.add_argument("--predictions", type=str, dest="predictions", required=False, default="")
    parser.add_argument("--prediction_probabilities", type=str, dest="prediction_probabilities",
                        required=False, default="")
    parser.add_argument("--output", type=str, dest="output")
    parser.add_argument("--config-file-name", dest="config_file_name", required=False, type=str, default="")
    parser.add_argument("--ground_truths_mltable", type=str, dest="ground_truths_mltable",
                        required=False, default="")
    parser.add_argument("--predictions_mltable", type=str, dest="predictions_mltable", required=False, default="")
    parser.add_argument("--prediction_probabilities_mltable", type=str, dest="prediction_probabilities_mltable",
                        required=False, default="")
    parser.add_argument("--ground_truths_column_name", type=str, dest="ground_truths_column_name",
                        required=False, default=None)
    parser.add_argument("--predictions_column_name", type=str, dest="predictions_column_name", 
                        required=False, default=None)
    args = parser.parse_args()
    print(args)

    custom_dimensions.app_name = constants.TelemetryConstants.COMPUTE_METRICS_NAME
    custom_dims_dict = vars(custom_dimensions)
    with log_activity(logger, constants.TelemetryConstants.COMPUTE_METRICS_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments")
        with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                          custom_dimensions=custom_dims_dict):
            validate_compute_metrics_args(args)

        is_ground_truths_mltable, ground_truths = check_and_return_if_mltable(args.ground_truths,
                                                                              args.ground_truths_mltable)
        is_predictions_mltable, predictions = check_and_return_if_mltable(args.predictions,
                                                                          args.predictions_mltable)
        is_prediction_probabilities_mltable, prediction_probabilities = check_and_return_if_mltable(
            args.prediction_probabilities, args.prediction_probabilities_mltable
        )

        runner = ModelEvaluationRunner(
            task=args.task,
            ground_truth=ground_truths,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            output=args.output,
            custom_dimensions=custom_dims_dict,
            config_file=args.config_file_name,
            is_ground_truth_mltable=is_ground_truths_mltable,
            is_predictions_mltable=is_predictions_mltable,
            is_prediction_probabilities_mltable=is_prediction_probabilities_mltable,
            ground_truths_column_name=args.ground_truths_column_name,
            predictions_column_name=args.predictions_column_name
        )
        runner.run()
    test_run.add_properties(properties=constants.RUN_PROPERTIES)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ == "__main__":
    test_component()
