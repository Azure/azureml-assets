# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main script for compute metrics."""
import azureml.evaluate.mlflow as aml_mlflow
import json
from azureml.telemetry.activity import log_activity

import constants
from constants import ArgumentLiterals, ForecastingConfigContract
from exceptions import (
    DataLoaderException,
    DataSavingException,
    DataValidationException,
    ComputeMetricsException,
)
from error_definitions import (
    ComputeMetricsInternalError,
    BadForecastData,
    InvalidGroundTruthColumnNameData,
    InvalidPredictionColumnNameData,
    BadInputData,
    SavingOutputError,
)
from logging_utilities import (
    custom_dimensions, current_run, get_logger, flush_logger,
    log_traceback, swallow_all_exceptions, get_azureml_exception
)
from utils import (
    ArgumentParser,
    fetch_compute_metrics_args,
    check_and_return_if_mltable,
    read_data,
    read_multiple_files,
    evaluate_predictions,
    parse_input_ground_truth_col,
    openai_init
)
from validation import validate_common_args, validate_compute_metrics_args

from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
import os
import pandas as pd
import numpy as np

custom_dimensions.app_name = constants.TelemetryConstants.COMPUTE_METRICS_NAME
logger = get_logger(name=__name__)
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
                 predictions_column_name: str = None,
                 extra_y_test_cols: str = None,
                 llm_config: dict = {}):
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
        self.ground_truths_column_name = ground_truths_column_name
        self.extra_y_test_cols = extra_y_test_cols
        self.predictions_column_name = predictions_column_name

        self.config = config
        self._is_multilabel = self.config.get("multilabel", False)
        self._has_multiple_output = self._is_multilabel or self.task == constants.TASK.NER
        if self.task == constants.TASK.QnA:
            for k, v in constants.OpenAIConstants.DEFAULT_OPENAI_CONFIG.items():
                if k not in llm_config:
                    logger.info(f"Required Key '{k}' not found in openai_config_params. \
                                Setting it to default '{v}'")
                    llm_config[k] = v
            do_openai_init = True
            self.qna_input_data_keys = {}
            for k in constants.OpenAIConstants.REQUIRED_KEYS:
                if k not in llm_config:
                    logger.warning(f"Required Key '{k}' missing in openai_config_params.\n\
                                Skipping GPT Based Metrics calculcation for this Run.")
                    do_openai_init = False
                    break
                else:
                    self.qna_input_data_keys[k] = llm_config[k]
            if do_openai_init:
                openai_init_params = {}
                for k, v in constants.OpenAIConstants.DEFAULT_OPENAI_INIT_PARAMS.items():
                    openai_init_params[k] = llm_config.pop(k, v)
                openai_params = openai_init(llm_config, **openai_init_params)
                if len(openai_params):
                    self.config[constants.OpenAIConstants.METRICS_KEY] = openai_params

    def load_data(self):
        """Load Test data.

        Returns:
            _type_: _description_
        """
        ground_truth = None
        if self.ground_truth:
            if os.path.isdir(self.ground_truth) and not self.is_ground_truth_mltable:
                ground_truth = read_multiple_files(self.ground_truth)
            else:
                ground_truth = read_data(self.ground_truth)
            ground_truth = list(ground_truth)[0]

            if self.config.get(constants.OpenAIConstants.METRICS_KEY):
                questions_key = self.qna_input_data_keys[constants.OpenAIConstants.QUESTIONS_KEY]
                contexts_key = self.qna_input_data_keys[constants.OpenAIConstants.CONTEXTS_KEY]
                questions, contexts = fetch_questions_and_contexts(ground_truth, questions_key,
                                                                   contexts_key, self.ground_truths_column_name)
                if len(questions) and len(contexts):
                    self.config[constants.OpenAIConstants.QUESTIONS_KEY] = questions
                    self.config[constants.OpenAIConstants.CONTEXTS_KEY] = contexts
                else:
                    logger.warning("Failed to Fetch Questions and Contexts from Ground Truth Data.\n\
                                   Skipping GPT Based Metrics Calculation")
                    self.config.pop(constants.OpenAIConstants.METRICS_KEY)

            if len(ground_truth) > 0:
                ground_truth = filter_ground_truths(ground_truth, self.task, self.ground_truths_column_name,
                                                    self.extra_y_test_cols, self.config)

        if os.path.isdir(self.predictions) and not self.is_predictions_mltable:
            predictions = read_multiple_files(path=self.predictions)
        else:
            predictions = read_data(self.predictions)
        predictions = list(predictions)[0]
        if self.predictions_column_name is not None:
            predictions = filter_predictions(predictions, self.task, self.predictions_column_name)

        predictions_probabilities = None
        if self.predictions_probabilities is not None:
            if os.path.isdir(self.predictions_probabilities) and not self.is_predictions_probabilities_mltable:
                predictions_probabilities = read_multiple_files(path=self.predictions_probabilities)
            else:
                predictions_probabilities = read_data(self.predictions_probabilities)
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
            exception = get_azureml_exception(ComputeMetricsException, ComputeMetricsInternalError, e,
                                              wrap_azureml_ex=False, error=repr(e))
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


def fetch_questions_and_contexts(data, questions_key, contexts_key, gt_column_name=None):
    """Fetch Questions and Contexts from QnA Dataset.

    Args:
        data (_type_): _description_
        questions_key (_type_): _description_
        contexts_key (_type_): _description_
    """
    questions, contexts = [], []
    if len(data.columns) > 1:
        if questions_key in data.columns and contexts_key in data.columns:
            questions = data[questions_key].tolist()
            contexts = data[contexts_key].tolist()
        elif gt_column_name is not None and gt_column_name in data.columns:
            if not isinstance(data[gt_column_name][0], dict):
                logger.warning("To calculate GPT based metrics, Questions and Contexts are required to be a \
                            part of Ground Truths. Ground Truths should be in SQuAD format.\n\
                            Skipping GPT Based metrics calculation for this Run.")
                return [], []
            sample = data[gt_column_name][0]
            if questions_key not in sample or contexts_key not in sample:
                logger.warning("Either Contexts Key or Questions Key not found in the Ground Truths Data row.\n\
                               Skipping GPT Based Metrics Calculation for this Run.")
                return [], []
            questions = data[gt_column_name].apply(lambda x: x[questions_key]).tolist()
            contexts = data[gt_column_name].apply(lambda x: x[contexts_key]).tolist()

    else:
        if not isinstance(data.tolist()[0], dict):
            logger.warning("To calculate GPT based metrics, Questions and Contexts are required to be a \
                            part of Ground Truths. Ground Truths should be in SQuAD format.\n\
                            Skipping GPT Based metrics calculation for this Run.")
            return [], []
        sample = data.tolist()[0]
        if questions_key not in sample or contexts_key not in sample:
            logger.warning("Either Contexts Key or Questions Key not found in the Ground Truths Data row.\n\
                            Skipping GPT Based Metrics Calculation for this Run.")
            return [], []
        questions = data[gt_column_name].apply(lambda x: x[questions_key]).tolist()
        contexts = data[gt_column_name].apply(lambda x: x[contexts_key]).tolist()
    return questions, contexts


def filter_ground_truths(data, task_type, column_name=None, extra_y_test_cols=None, config=None):
    """Read Json file utility function.

    Args:
        data (_type_): _description_
        column_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if task_type in [constants.TASK.IMAGE_INSTANCE_SEGMENTATION, constants.TASK.IMAGE_OBJECT_DETECTION,
                     constants.TASK.FORECASTING] or \
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
                exception = get_azureml_exception(DataValidationException, InvalidGroundTruthColumnNameData, e)
                log_traceback(exception, logger)
                raise exception
        if column_name in data.columns:
            if isinstance(data[column_name].iloc[0], list) or isinstance(data[column_name].iloc[0], np.ndarray):
                logger.warning("Multiple ground truths are not supported for the Question and Answering currently.\
                                Considering only the first ground truth in case of multiple values.")
                data[column_name] = data[column_name].apply(lambda x: x[0])
    if len(data.columns) > 1:
        label_cols = []
        if column_name:
            label_cols += [column_name]
        if extra_y_test_cols:
            label_cols += extra_y_test_cols
        if len(label_cols) > 0:
            if all(col in data.columns for col in label_cols):
                data = data[label_cols]
            else:
                exception = get_azureml_exception(DataValidationException, InvalidGroundTruthColumnNameData, None)
                log_traceback(exception, logger)
                raise exception
        else:
            logger.warning("Multiple columns found in ground truths. Taking all into consideration.")
            # exception = get_azureml_exception(DataValidationException, InvalidGroundTruthColumnName, None)
            # log_traceback(exception, logger)
            # raise exception
    return data


def filter_predictions(data, task_type, column_name):
    """Read Json file utility function.

    Args:
        data (_type_): _description_
        column_name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if task_type in [constants.TASK.CHAT_COMPLETION]:
        # do not filter as these contains multiple prediction columns
        return data
    #  for Question-Answering checking for multiple columns in ground truth
    if task_type == constants.TASK.QnA:
        if not column_name:
            column_name = constants.PREDICTIONS_COLUMN_NAME
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
                message_kwargs = {"prediction_column": column_name}
                exception = get_azureml_exception(DataValidationException, InvalidPredictionColumnNameData, e,
                                                  **message_kwargs)
                log_traceback(exception, logger)
                raise exception
        if column_name in data.columns:
            if isinstance(data[column_name].iloc[0], list) or isinstance(data[column_name].iloc[0], np.ndarray):
                logger.warning("Multiple predictions are not supported for the Question and Answering currently.\
                                Considering only the first prediction in case of multiple values.")
                data[column_name] = data[column_name].apply(lambda x: x[0])
    if len(data.columns) > 1:
        if column_name and column_name in data.columns:
            logger.info("Multiple columns found. Picking only prediction column.")
            try:
                data = data[[column_name]]
            except Exception as e:
                message_kwargs = {"prediction_column": column_name}
                exception = get_azureml_exception(DataValidationException, InvalidPredictionColumnNameData, e,
                                                  **message_kwargs)
                log_traceback(exception, logger)
                # raise exception
                logger.info("Error in filtering prediction columns. Taking all into consideration.")
        else:
            logger.warning("Multiple columns found in predictions. Taking all into consideration.")
    return data


@swallow_all_exceptions(logger)
def run():
    """Entry point for compute metrics component."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest=ArgumentLiterals.TASK, choices=constants.ALL_TASKS)
    parser.add_argument("--ground_truths", type=str, dest=ArgumentLiterals.GROUND_TRUTHS, required=True)
    parser.add_argument("--ground_truths_column_name", type=lambda x: x.split(","),
                        dest=ArgumentLiterals.GROUND_TRUTHS_COLUMN_NAME, required=False, default=None)
    parser.add_argument("--predictions", type=str, dest=ArgumentLiterals.PREDICTIONS, required=True)
    parser.add_argument("--predictions_column_name", type=str,
                        dest=ArgumentLiterals.PREDICTIONS_COLUMN_NAME, required=False, default=None)
    parser.add_argument("--prediction_probabilities", type=str, dest=ArgumentLiterals.PREDICTION_PROBABILITIES,
                        required=False, default="")
    parser.add_argument("--config_str", type=str, dest=ArgumentLiterals.CONFIG_STR, required=False, default=None)
    parser.add_argument("--config-file-name", type=str, dest=ArgumentLiterals.CONFIG_FILE_NAME,
                        required=False, default="")
    parser.add_argument("--openai-config-params", type=str, dest="openai_config_params", required=False, default="{}")

    # Outputs
    parser.add_argument("--output", type=str, dest=ArgumentLiterals.OUTPUT)

    args, _ = parser.parse_known_args()
    args = vars(args)

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args))
        validate_common_args(args)
        validate_compute_metrics_args(args)
        config = fetch_compute_metrics_args(args[ArgumentLiterals.CONFIG], args[ArgumentLiterals.TASK])

        ground_truths_column_name, extra_y_test_cols = parse_input_ground_truth_col(
            args[ArgumentLiterals.GROUND_TRUTHS_COLUMN_NAME]
        )

        ground_truths = args[ArgumentLiterals.GROUND_TRUTHS]
        is_ground_truths_mltable = check_and_return_if_mltable(ground_truths)
        predictions = args[ArgumentLiterals.PREDICTIONS]
        is_predictions_mltable = check_and_return_if_mltable(predictions)
        prediction_probabilities = args[ArgumentLiterals.PREDICTION_PROBABILITIES]
        is_prediction_probabilities_mltable = check_and_return_if_mltable(
            args[ArgumentLiterals.PREDICTION_PROBABILITIES]
        )

        if args[ArgumentLiterals.TASK] == constants.TASK.FORECASTING:
            if not ground_truths_column_name:
                # If the ground true column name was not provided, we will try to take it from the config.
                ground_truths_column_name = config.get(ForecastingConfigContract.FORECAST_GROUND_TRUTH, '')
            if not args[ArgumentLiterals.PREDICTIONS_COLUMN_NAME]:
                args[ArgumentLiterals.PREDICTIONS_COLUMN_NAME] = config.get(
                    ForecastingConfigContract.FORECAST_PREDICTIONS, '')
            if not ground_truths_column_name or (
                    not is_ground_truths_mltable and not args[ArgumentLiterals.GROUND_TRUTHS]
            ):
                exception = get_azureml_exception(DataValidationException, BadForecastData, None)
                log_traceback(exception, logger)
                raise exception
        llm_config = {}
        try:
            llm_config = json.loads(args.openai_config_params)
            if not isinstance(llm_config, dict):
                raise ValueError(f"Openai config is expected to be a json encoded dictionary. \
                                 Got {type(llm_config)} instead")
        except Exception as e:
            logger.warning(f"Loading OpenAI Config Params Failed with exception {repr(e)}.\n\
                           Skipping GPT Based Metrics Calculation.")

    logger.info(f"LLM Config {llm_config}")
    with log_activity(logger, constants.TelemetryConstants.INITIALISING_RUNNER,
                      custom_dimensions=custom_dims_dict):
        runner = ComputeMetricsRunner(
            task=args[ArgumentLiterals.TASK],
            ground_truth=ground_truths,
            predictions=predictions,
            prediction_probabilities=prediction_probabilities,
            output=args[ArgumentLiterals.OUTPUT],
            config=config,
            is_ground_truth_mltable=is_ground_truths_mltable,
            is_predictions_mltable=is_predictions_mltable,
            is_prediction_probabilities_mltable=is_prediction_probabilities_mltable,
            ground_truths_column_name=ground_truths_column_name,
            predictions_column_name=args[ArgumentLiterals.PREDICTIONS_COLUMN_NAME],
            extra_y_test_cols=extra_y_test_cols,
            llm_config=llm_config
        )

    with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        try:
            logger.info("Loading Data.")
            flush_logger(logger)
            runner.load_data()
        except Exception as e:
            exception = get_azureml_exception(DataLoaderException, BadInputData, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception

    with log_activity(logger, activity_name=constants.TelemetryConstants.COMPUTE_METRICS_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Computing metrics.")
        result = runner.compute_metrics()

    with log_activity(logger, activity_name=constants.TelemetryConstants.LOG_AND_SAVE_OUTPUT,
                      custom_dimensions=custom_dims_dict):
        logger.info("Logging and Saving outputs.")
        try:
            runner.log_and_write_outputs(result)
        except Exception as e:
            exception = get_azureml_exception(DataSavingException, SavingOutputError, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception

    test_run.add_properties(properties=constants.RUN_PROPERTIES)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ == "__main__":
    run()
