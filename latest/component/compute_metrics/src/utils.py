# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model Evaluation utilities."""
import constants
import numpy as np
import pandas as pd
import pickle
import os
import sys
import argparse
import json
import openai
import glob

from constants import TASK, ForecastingConfigContract, ArgumentLiterals
from workspace_utils import (get_connection_by_id_v2,
                             workspace_connection_to_credential,
                             get_target_from_connection,
                             get_metadata_from_connection)
from logging_utilities import get_logger, log_traceback
from mltable import load
from task_factory.tabular.classification import TabularClassifier
from task_factory.text.classification import TextClassifier
from task_factory.tabular.regression import TabularRegressor
from task_factory.tabular.forecast import TabularForecast
# from task_factory.text.regression import TextRegressor
from task_factory.text.ner import TextNerPredictor
from task_factory.text.qna import QnAPredictor
from task_factory.text.summarization import Summarizer
from task_factory.text.translation import Translator
from task_factory.text.text_generation import TextGenerator
from task_factory.text.fill_mask import FillMask
from task_factory.text.chat_completion import ChatCompletion
from task_factory.image.classification import ImageMulticlassClassifier, ImageMultilabelClassifier
from task_factory.image.od_is import ImageOdIsPredictor
from evaluators.evaluators import EvaluatorFactory
from logging_utilities import current_run, get_azureml_exception
from azureml.metrics import _scoring_utilities, constants as metrics_constants
from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
import azureml.evaluate.mlflow as aml_mlflow
from azureml.evaluate.mlflow.models.evaluation import EvaluationResult
from run_utils import TestRun
from image_constants import ImageDataFrameParams
from error_definitions import (
    BadRegressionData,
    MetricsLoggingError,
    BadInputData,
    ArgumentParsingError,
    BadEvaluationConfigFile,
    BadEvaluationConfigParam,
)
from exceptions import (
    DataValidationException,
    DataLoaderException,
    ComputeMetricsException,
    ModelEvaluationException,
    ArgumentValidationException,
)

logger = get_logger(name=__name__)


class ArgumentParser(argparse.ArgumentParser):
    """Model Evaluation Custom Argument.

    Args:
        argparse.ArgumentParser (_type_): _description_
    """

    def error(self, message):
        """Override ArgumentParser internal error call."""
        self.print_usage(sys.stderr)
        args = {'prog': self.prog, 'message': message}
        message = '%(prog)s: error: %(message)s' % args
        exception = get_azureml_exception(ArgumentValidationException, ArgumentParsingError, None, error=message)
        log_traceback(exception, logger)
        raise exception


def assert_and_raise(condition, exception_cls, error_cls, **message_kwargs):
    """Assert condition and raise the error if false.

    Args:
        condition (_type_): _description_
        exception_cls (_type_): _description_

    Raises:
        exception: _description_
    """
    if not condition:
        exception = get_azureml_exception(exception_cls, error_cls, None, **message_kwargs)
        log_traceback(exception, logger)
        raise exception


def filter_pipeline_params(evaluation_config):
    """Filter Pipeline params in evaluation_config.

    Args:
        evaluation_config (_type_): _description_

    Returns:
        _type_: _description_
    """
    filtered_params = {i: j for i, j in evaluation_config.items() if i in constants.ALLOWED_PIPELINE_PARAMS}
    return filtered_params


def load_transformer(filename):
    """Load y transformer.

    Args:
        filename (_type_): _description_

    Raises:
        DataValidationException: _description_

    Returns:
        _type_: _description_
    """
    try:
        with open(filename, "rb") as f:
            y_transformer = pickle.load(f)
    except Exception as e:
        error_message = "Not able to load y_transformer. Check file format."
        exception = get_azureml_exception(DataValidationException, BadEvaluationConfigParam, e, error=repr(e))
        log_traceback(exception, logger, error_message)
        raise exception
    return y_transformer


def _log_metrics(metrics, artifacts):
    """Log metrics and artifacts to current run.

    Args:
        metrics (_type_): _description_
        artifacts (_type_): _description_

    Raises:
        ModelEvaluationException: _description_
    """
    table_scores = {}
    nonscalar_scores = {}
    list_metrics = [metrics_constants.Metric.FMPerplexity]
    run = current_run.run
    list_scores = {}
    classwise_scores = {}

    for name, score in artifacts.items():
        if score is None:
            logger.warning("Empty score for {}. Skipping.".format(name))
            continue
        elif _scoring_utilities.is_table_metric(name) or name in metrics_constants.Metric.QA_GPT_METRICS_SET \
                or name == metrics_constants.Metric.BERTScore:
            table_scores[name] = score
        elif name in list_metrics:
            try:
                list_scores[name] = list(score)
                if name == metrics_constants.Metric.FMPerplexity:
                    metrics["mean_" + name] = np.mean(score)
            except TypeError:
                logger.warning(f"{name} is not of type list.")
        elif name in metrics_constants.Metric.NONSCALAR_FULL_SET or \
                name in metrics_constants.FULL_NONSCALAR_SET:
            nonscalar_scores[name] = score
        elif name in metrics_constants.TrainingResultsType.ALL_TIME:
            # Filter out time metrics as we do not log these
            pass
        elif name in metrics_constants.FULL_CLASSWISE_SET:
            classwise_scores[name] = score
        else:
            logger.warning("Unknown metric {}. Will not log.".format(name))

    try:
        # Log the scalar metrics. (Currently, these are stored in CosmosDB)
        for name, score in metrics.items():
            if isinstance(score, list):
                list_scores[name] = list(score)
                continue
            run.log(name, score)

        for name, score in table_scores.items():
            if name == metrics_constants.Metric.BERTScore:
                for k, v in score.items():
                    if not isinstance(v, np.ndarray) and not isinstance(v, list):
                        continue
                    x, y = np.histogram(v, bins=10)
                    run.log_table("Bert F1 Score", value={"_score": list(y)[1:], "count": list(x)})

                    x, y = np.histogram(v, bins=10)
                    run.log_table("Bert Precision", value={"_score": list(y)[1:], "count": list(x)})

                    x, y = np.histogram(v, bins=10)
                    run.log_table("Bert Recall", value={"_score": list(y)[1:], "count": list(x)})
            elif name in metrics_constants.Metric.QA_GPT_METRICS_SET:
                try:
                    if not isinstance(score, list) and not isinstance(score, np.ndarray):
                        logger.warning(f"{name} is not an iterable. \nValue: {score}")
                        continue
                    int_score = [int(i) for i in score]
                    counts = [0]*5
                    for i in int_score:
                        counts[i-1] += 1
                    cur_score = {
                        "_rating": [i for i in range(1, 6)],
                        "count": counts
                    }
                    run.log_table(name, cur_score)
                except Exception as e:
                    if (isinstance(score, list) or isinstance(score, np.ndarray)) and len(score) > 0:
                        exception_cls_name = score[0]
                        logger.warning(f"Ignoring metric: {name}\n Computation Failed due to: {exception_cls_name}")
                    else:
                        logger.warning(f"Ignoring metric: {name}\n Logging Failed due to: {repr(e)}")
            else:
                run.log_table(name, score)

        # for name, score in list_scores.items():
        #     # TODO: Add checks for logging longer lists
        #     pass
        #     # run.log_list(name, score)

        # Log the non-scalar metrics. (Currently, these are all artifact-based.)
        for name, score in nonscalar_scores.items():
            if name == metrics_constants.Metric.AccuracyTable:
                run.log_accuracy_table(name, score)
            elif name in metrics_constants.Metric.IMAGE_LEVEL_BINARY_CLASSIFIER_METRICS:
                run.log_table(name, score)
            elif name == metrics_constants.Metric.ConfusionMatrix:
                run.log_confusion_matrix(name, score)
            elif name == metrics_constants.Metric.CONFUSION_MATRICES_PER_SCORE_THRESHOLD:
                for key, confusion_matrix in score.items():
                    run.log_confusion_matrix(key, confusion_matrix)
            elif name == metrics_constants.Metric.Residuals:
                run.log_residuals(name, score)
            elif name == metrics_constants.Metric.PredictedTrue:
                run.log_predictions(name, score)
            elif name in metrics_constants.Metric.NONSCALAR_FORECAST_SET:
                # Filter out non-scalar forecasting metrics as we do not log these yet
                pass
            else:
                logger.warning("Unsupported non-scalar metric {}. Will not log.".format(name))

        # Log the classwise metrics. (Currently, these are all artifact-based.)
        for name, score in classwise_scores.items():
            try:
                if name == metrics_constants.Metric.PER_LABEL_METRICS:
                    for metrics in score.values():
                        run.log_table(name, metrics)
                else:
                    logger.warning("Unsupported non-scalar metric {}. Will not log.".format(name))
            except Exception:  # TODO
                e = ModelEvaluationException(f"Failed to log classwise metric {name} with value {score}")
                log_traceback(e, logger)
                raise e
    except Exception as e:
        exception = get_azureml_exception(ComputeMetricsException, MetricsLoggingError, e, wrap_azureml_ex=False,
                                          metric_name=name, error=repr(e))
        log_traceback(exception, logger)
        raise exception


def evaluate_predictions(y_test, y_pred, y_pred_proba, task_type, metrics_config, X_test=None,
                         input_column_names=None, is_multiple_ground_truth=False, **kwargs):
    """Compute metrics mode method.

    Args:
        y_test (_type_): _description_
        y_pred (_type_): _description_
        y_pred (_type_) : _description_
        task_type (_type_): _description_
        metrics_config (_type_): _description_
        X_test (_type_): _description_

    Returns:
        _type_: _description_
    """
    evaluator = EvaluatorFactory().get_evaluator(task_type, metrics_config)
    res = evaluator.evaluate(y_test, y_pred, y_pred_proba=y_pred_proba, X_test=X_test,
                             input_column_names=input_column_names,
                             is_multiple_ground_truth=is_multiple_ground_truth, **kwargs)
    metrics = res[metrics_constants.Metric.Metrics]
    artifacts = res[metrics_constants.Metric.Artifacts]
    _log_metrics(metrics, artifacts)
    keys = artifacts.keys()
    for k in keys:
        json_content = artifacts[k]
        json_artifact = JsonEvaluationArtifact(uri=aml_mlflow.get_artifact_uri(k), content=json_content)
        artifacts[k] = json_artifact
    result = EvaluationResult(metrics=metrics, artifacts=artifacts)

    return result


def get_predictor(task):
    """Get predictor.

    Args:
        task (_type_): _description_

    Returns:
        _type_: _description_
    """
    predictor_map = {
        TASK.CLASSIFICATION: TabularClassifier,
        TASK.CLASSIFICATION_MULTILABEL: TabularClassifier,
        TASK.TEXT_CLASSIFICATION: TextClassifier,
        TASK.TEXT_CLASSIFICATION_MULTILABEL: TextClassifier,
        TASK.REGRESSION: TabularRegressor,
        TASK.NER: TextNerPredictor,
        TASK.QnA: QnAPredictor,
        TASK.SUMMARIZATION: Summarizer,
        TASK.TRANSLATION: Translator,
        TASK.TEXT_GENERATION: TextGenerator,
        TASK.FILL_MASK: FillMask,
        TASK.CHAT_COMPLETION: ChatCompletion,
        TASK.FORECASTING: TabularForecast,
        TASK.IMAGE_CLASSIFICATION: ImageMulticlassClassifier,
        TASK.IMAGE_CLASSIFICATION_MULTILABEL: ImageMultilabelClassifier,
        TASK.IMAGE_OBJECT_DETECTION: ImageOdIsPredictor,
        TASK.IMAGE_INSTANCE_SEGMENTATION: ImageOdIsPredictor
    }
    return predictor_map.get(task)


class ArgumentsSet:
    """Argument Set fetcher for given tasks."""

    def __init__(self, task_type) -> None:
        """__init__.

        Args:
            task_type (_type_): _description_
        """
        self.args_set = {}
        if task_type in constants.CLASSIFICATION_SET:
            self.args_set = self.classification
        if task_type == TASK.REGRESSION:
            self.args_set = self.regression
        if task_type == TASK.NER:
            self.args_set = self.text_ner
        if task_type == TASK.SUMMARIZATION:
            self.args_set = self.text_summarization
        if task_type == TASK.TRANSLATION:
            self.args_set = self.text_translation
        if task_type == TASK.QnA:
            self.args_set = self.qna
        if task_type == TASK.TEXT_GENERATION:
            self.args_set = self.text_generation
        if task_type == TASK.FILL_MASK:
            self.args_set = self.fill_mask
        if task_type == TASK.FORECASTING:
            self.args_set = self.forecasting
        if task_type in [TASK.IMAGE_OBJECT_DETECTION, TASK.IMAGE_INSTANCE_SEGMENTATION]:
            self.args_set = self.image_od_is
        if task_type == TASK.IMAGE_CLASSIFICATION:
            self.args_set = self.image_classification
        if task_type == TASK.IMAGE_CLASSIFICATION_MULTILABEL:
            self.args_set = self.image_classification_multilabel

    @property
    def classification(self):
        """Classification arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "class_labels": "np.asarray(val)",
            "train_labels": "np.asarray(val)",
            "sample_weight": "np.asarray(val)",
            "y_transformer": "load_transformer(val)",
            "use_binary": "bool(val)",
            "enable_metric_confidence": "bool(val)",
            "multilabel": "bool(val)",
            "positive_label": "pass",
            "confidence_metrics": "list(val)"
        }
        return args_map

    @property
    def regression(self):
        """Regression Arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "y_max": "np.asarray(val)",
            "y_min": "np.asarray(val)",
            "y_std": "np.asarray(val)",
            "bin_info": "dict(val)",
            "sample_weight": "np.asarray(val)",
            "enable_metric_confidence": "bool(val)",
            "confidence_metrics": "list(val)"
        }
        return args_map

    @property
    def text_ner(self):
        """Text NER arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "train_label_list": "list(val)"
        }
        return args_map

    @property
    def text_summarization(self):
        """Text Summarization arguments.

        Return:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "aggregator": "bool(val)",
            "stemmer": "bool(val)"
        }
        return args_map

    @property
    def text_translation(self):
        """Text Translation arguments.

        Return:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "smoothing": "bool(val)",
            "source_lang": "str(val)",
            "target_lang": "str(val)"
        }
        return args_map

    @property
    def text_generation(self):
        """Text Generation arguments.

        Return:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "smoothing": "bool(val)",
            "aggregator": "bool(val)",
            "stemmer": "bool(val)",
            "sub_task": "str(val)",
            "allow_code_eval": "bool(val)",
            "no_of_candidates": "list(val)",
            "num_workers": "int(val)",
            "timeout": "int(val)"
        }
        return args_map

    @property
    def qna(self):
        """Question Answering arguments.

        Return:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "regexes_to_ignore": "list(val)",
            "ignore_case": "bool(val)",
            "ignore_punctuation": "bool(val)",
            "ignore_numbers": "bool(val)"
        }
        return args_map

    @property
    def fill_mask(self):
        """Fill Masking arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "model_id": "str(val)",
            "batch_size": "int(val)",
            "add_start_token": "bool(val)"
        }
        return args_map

    @property
    def forecasting(self):
        """Forecasting arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            ForecastingConfigContract.TIME_COLUMN_NAME: "str(val)",
            ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: "val",
            ForecastingConfigContract.FORECAST_FLAVOR: "str(val)",
            ForecastingConfigContract.ROLLING_FORECAST_STEP: "int(val)",
            ForecastingConfigContract.FORECAST_ORIGIN_COLUMN_NAME: "str(val)",
            ForecastingConfigContract.FORECAST_PREDICTIONS: "val",
            ForecastingConfigContract.FORECAST_GROUND_TRUTH: "val"
        }
        return args_map

    @property
    def image_od_is(self):
        """Image OD/IS arguments.

        Returns:
            _type_: Dictionary of arguments.
        """
        args_map = {
            "metrics": "list(val)",
            "iou_threshold": "float(val)",
            "box_score_threshold": "float(val)",
        }
        return args_map

    @property
    def image_classification(self):
        """Image Classification arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
        }
        return args_map

    @property
    def image_classification_multilabel(self):
        """Image Classification Multilabel arguments.

        Returns:
            _type_: _description_
        """
        args_map = {
            "metrics": "list(val)",
            "threshold": "float(val)",
        }
        return args_map


def check_and_return_if_mltable(data):
    """Is current director MLTable or not.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_mltable = False
    if os.path.isdir(data):
        local_yaml_path = os.path.join(data, constants.MLTABLE_FILE_NAME)
        if os.path.exists(local_yaml_path):
            is_mltable = True
    return is_mltable


def read_model_prediction_data(file_path, task=None, batch_size=None, nrows=None):
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
    if task in constants.IMAGE_TASKS:
        from image_dataset import get_image_dataset
        df = get_image_dataset(task_type=task, test_mltable=file_path)
        data = iter([df])
    else:
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
        logger.warn("Received URI_FOLDER instead of URI_FILE. Checking if part of LLM Pipeline")
        if constants.LLM_FT_PREPROCESS_FILENAME not in os.listdir(file_path):
            message = "Test Data is a folder. JSON Lines File or MLTable expected."
            exception = get_azureml_exception(DataLoaderException, BadInputData, None, error=message)
            log_traceback(exception, logger, message)
            raise exception
        logger.info("Found LLM Preprocess args")
        with open(os.path.join(file_path, constants.LLM_FT_PREPROCESS_FILENAME)) as f:
            llm_preprocess_args = json.load(f)
        test_data_path = llm_preprocess_args[constants.LLM_FT_TEST_DATA_KEY]
        file_path = os.path.join(file_path, test_data_path)
    try:
        if is_mltable:
            if batch_size:
                logger.warning("batch_size not supported with MLTable files. Ignoring parameter.")
            mltable_data = load(file_path)
            if nrows:
                mltable_data = mltable_data.take(nrows)
            data = iter([mltable_data.to_pandas_dataframe()])
        else:
            data = read_dataframe(file_path, batch_size=batch_size, nrows=nrows)
            if not batch_size:
                data = iter([data])
    except Exception as e:
        exception = get_azureml_exception(DataLoaderException, BadInputData, e, error=repr(e))
        log_traceback(exception, logger)
        raise exception
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
        exception = get_azureml_exception(DataLoaderException, BadInputData, None, error=error)
        log_traceback(exception, logger)
        raise exception
    data = pd.concat(dfs, ignore_index=True)
    return iter([data])


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
    if task == constants.TASK.REGRESSION:
        if y_test is not None:
            try:
                y_test = y_test.astype(np.float64)
            except Exception as e:
                exception = get_azureml_exception(DataLoaderException, BadRegressionData, e,
                                                  error=repr(e), y_test_dtype=y_test.dtype)
                log_traceback(exception, logger)
                raise exception
    if task == constants.TASK.NER:
        if len(X_test.columns) > 1 and "tokens" not in X_test.columns:
            message = "Too many feature columns in dataset. Only 1 feature column should be passed for NER."
            exception = get_azureml_exception(DataLoaderException, BadInputData, None, error=message)
            log_traceback(exception, logger)
            raise exception
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

    if task == constants.TASK.QnA and y_test is not None:
        if isinstance(y_test.iloc[0], dict):
            # Extracting only the first one for now
            # TODO: Fix this post PrP
            y_test = y_test.apply(lambda x: x["text"][0] if len(x["text"]) > 0 else "")
        elif isinstance(y_test.iloc[0], list) or isinstance(y_test.iloc[0], np.ndarray):
            y_test = y_test.apply(lambda x: x[0])
        if not isinstance(y_test.iloc[0], str):
            message = "Ground Truths for Question-answering should be a string or an array. " \
                      "Found: " + type(y_test.iloc[0])
            exception = get_azureml_exception(DataLoaderException, BadInputData, None, error=message)
            log_traceback(exception, logger, message)
            raise exception
    if task == constants.TASK.FILL_MASK and y_test is not None:
        if isinstance(y_test.iloc[0], np.ndarray) or isinstance(y_test.iloc[0], list):
            y_test = y_test.apply(lambda x: tuple(x))
        if not isinstance(y_test.iloc[0], str) and not isinstance(y_test.iloc[0], tuple):
            message = "Ground Truths for Fill-Mask should be a string or an array found " + type(y_test.iloc[0])
            exception = get_azureml_exception(DataLoaderException, BadInputData, None, error=message)
            log_traceback(exception, logger, message)
            raise exception

    if y_test is not None:
        y_test = y_test.values

    return X_test, y_test


def fetch_compute_metrics_args(data, task_type):
    """Fetch compute metrics arguments from evaluation config.

    Args:
        data : _description_
        task_type: _description_
    """
    metrics_args = ArgumentsSet(task_type=task_type)
    for arg, func in metrics_args.args_set.items():
        val = data.get(arg, None)
        if val is not None:
            try:
                data[arg] = eval(func)
            except TypeError as e:
                message = "Invalid dtype passed for config param '" + arg + "'."
                exception = get_azureml_exception(DataLoaderException, BadEvaluationConfigParam, e, error=repr(e))
                log_traceback(exception, logger, message)
                raise exception
            except Exception as e:
                exception = get_azureml_exception(DataLoaderException, BadEvaluationConfigParam, e, error=repr(e))
                log_traceback(exception, logger)
                raise exception
    return data


def read_config(conf_file):
    """Util function for reading config.

    Args:
        conf_file (_type_): _description_

    Raises:
        DataLoaderException: _description_

    Returns:
        _type_: _description_
    """
    if not conf_file:
        return dict()
    try:
        with open(conf_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        exception = get_azureml_exception(DataLoaderException, BadEvaluationConfigFile, e, error=repr(e))
        log_traceback(exception, logger)
        raise exception

    return data


def read_config_str(conf_str):
    """Util function for reading config string.

    Args:
        conf_str (_type_): _description_

    Raises:
        DataLoaderException: _description_

    Returns:
        _type_: _description_
    """
    if not conf_str:
        return dict()
    try:
        return json.loads(conf_str)
    except Exception as e:
        message = "Unable to load evaluation_config_params. String is not JSON serialized."
        exception = get_azureml_exception(DataLoaderException, BadEvaluationConfigParam, e, error=repr(e))
        log_traceback(exception, logger, message)
        raise exception


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


def get_column_names(args, data):
    """Get Column names from test data."""
    task = args[ArgumentLiterals.TASK]
    if task in constants.IMAGE_TASKS:
        input_column_names = [ImageDataFrameParams.IMAGE_COLUMN_NAME]
        label_column_name = ImageDataFrameParams.LABEL_COLUMN_NAME
        extra_y_test_cols = None
        if task in [constants.TASK.IMAGE_OBJECT_DETECTION, constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
            input_column_names.append(ImageDataFrameParams.IMAGE_META_INFO)
    else:
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


def openai_init(llm_config, **openai_params):
    """Initialize OpenAI Params."""
    logger.info(f"Using llm_config: {json.dumps(llm_config, indent=2)}")
    openai_api_type = openai_params.get("openai_api_type", "azure")
    openai_api_version = openai_params.get("openai_api_version", "2023-03-15-preview")

    connection_id = os.environ.get(constants.OpenAIConstants.CONNECTION_STRING_KEY, None)
    fetch_from_connection = False
    if connection_id is not None:
        connection = get_connection_by_id_v2(connection_id)
        credential = workspace_connection_to_credential(connection)
        if hasattr(credential, 'key'):
            llm_config["key"] = credential.key
            target = get_target_from_connection(connection)
            llm_config["base"] = target
            metadata = get_metadata_from_connection(connection)
            openai_api_type = metadata.get('apiType', "azure")
            openai_api_version = metadata.get('apiVersion', "2023-03-15-preview")
            logger.info("Using workspace connection key for OpenAI")
            fetch_from_connection = True
    if not fetch_from_connection:
        if llm_config.get("type") == "azure_open_ai":
            ws = TestRun().workspace
            keyvault = ws.get_default_keyvault()
            secrets = keyvault.get_secrets(secrets=[
                "BAKER-OPENAI-API-BASE",
                "BAKER-OPENAI-API-KEY",
                "OPENAI-API-KEY",
                "OPENAI-API-BASE"])
            logger.info("Run context and secrets retrieved")

            # hacky way to override OPENAI-API-KEY if Baker key existed
            if secrets["BAKER-OPENAI-API-BASE"] is not None:
                secrets["OPENAI-API-BASE"] = secrets["BAKER-OPENAI-API-BASE"]
            if secrets["BAKER-OPENAI-API-KEY"] is not None:
                secrets["OPENAI-API-KEY"] = secrets["BAKER-OPENAI-API-KEY"]

            if secrets["OPENAI-API-KEY"] is not None:
                llm_config["key"] = secrets["OPENAI-API-KEY"]
            if secrets["OPENAI-API-BASE"] is not None:
                llm_config["base"] = secrets["OPENAI-API-BASE"]
        else:
            logger.warn("No Connection String Provided and no credentials present in workspace's keyvault.")
            logger.warn("Skipping OpenAI Initialization.")
            return {}

    openai.api_version = openai_api_version
    openai.api_type = openai_api_type
    openai.api_base = llm_config.get("base", None)
    openai.api_key = llm_config.get("key", None)

    if not all([llm_config["base"], llm_config["key"], llm_config['deployment_name']]):
        logger.warn("No Connection String Provided and no credentials present in workspace's keyvault.")
        logger.warn("Skipping OpenAI Initialization.")
        return {}

    openai_final_params = {
        "api_version": openai_api_version,
        "api_type": openai_api_type,
        "api_base": llm_config["base"],
        "api_key": llm_config["key"],
        "deployment_id": llm_config['deployment_name']
    }
    return openai_final_params
