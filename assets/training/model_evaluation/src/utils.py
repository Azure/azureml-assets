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

from constants import TASK, ForecastingConfigContract
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
from logging_utilities import current_run
from azureml.metrics import _scoring_utilities, constants as metrics_constants
from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
import azureml.evaluate.mlflow as aml_mlflow
from azureml.evaluate.mlflow.models.evaluation import EvaluationResult
from azureml._common._error_definition.azureml_error import AzureMLError
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
from copy import deepcopy

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
        message = '%(prog)s: error: %(message)s\n' % args
        exception = ArgumentValidationException._with_error(
            AzureMLError.create(ArgumentParsingError, error=message),
        )
        log_traceback(exception, logger)
        raise exception


def assert_and_raise(condition, exception_cls, error_cls, message_kwargs={}):
    """Assert condition and raise the error if false.

    Args:
        condition (_type_): _description_
        exception_cls (_type_): _description_
        message (_type_): _description_

    Raises:
        exception: _description_
    """
    if not condition:
        exception = exception_cls._with_error(
            AzureMLError.create(error_cls, **message_kwargs)
        )
        log_traceback(exception, logger, is_critical=True)
        raise exception


def get_task_from_model(model_uri):
    """Get task type from model config.

    Args:
        model_uri (_type_): _description_

    Returns:
        _type_: _description_
    """
    mlflow_model = aml_mlflow.models.Model.load(model_uri)
    task_type = ""
    if "hftransformers" in mlflow_model.flavors:
        if "task_type" in mlflow_model.flavors["hftransformers"]:
            task_type = mlflow_model.flavors["hftransformers"]["task_type"]
            if task_type.startswith("translation"):
                task_type = constants.TASK.TRANSLATION
    return task_type


def filter_pipeline_params(evaluation_config):
    """Filter Pipeline params in evaluation_config.

    Args:
        evaluation_config (_type_): _description_

    Returns:
        _type_: _description_
    """
    filtered_params = {i: j for i, j in evaluation_config.items() if i in constants.ALLOWED_PIPELINE_PARAMS}
    return filtered_params


def sanitize_device_and_device_map(evaluation_config, device):
    """Check device_map and device in args.

    Args:
        evaluation_config (_type_): _description_
        device (_type_): _description_

    Returns:
        _type_: _description_
    """
    has_device_map_auto = False
    if evaluation_config.get("pipeline_init_args"):
        if evaluation_config["pipeline_init_args"].get("device_map"):
            has_device_map_auto = evaluation_config["pipeline_init_args"]["device_map"] == constants.DEVICE.AUTO
    if has_device_map_auto:
        return evaluation_config, None
    else:
        if device == constants.DEVICE.AUTO:
            new_evaluation_config = deepcopy(evaluation_config)
            if not new_evaluation_config.get("pipeline_init_args"):
                new_evaluation_config["pipeline_init_args"] = {}
            new_evaluation_config["pipeline_init_args"]["device_map"] = constants.DEVICE.AUTO
            return new_evaluation_config, None
        return evaluation_config, device


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
        error_message = "Not able to load y_transformer. Check file format"
        log_traceback(e, logger, error_message)
        raise DataValidationException(error_message)
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
            continue
        elif _scoring_utilities.is_table_metric(name):
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
            except Exception:
                e = ModelEvaluationException(f"Failed to log classwise metric {name} with value {score}")
                log_traceback(e, logger)
                raise e
    except Exception as e:
        if isinstance(e, ModelEvaluationException):
            raise e
        exception = ComputeMetricsException._with_error(
            AzureMLError.create(MetricsLoggingError, metric_name=name, error=repr(e)),
            inner_exception=e
        )
        log_traceback(exception, logger)
        raise exception


def evaluate_predictions(y_test, y_pred, y_pred_proba, task_type, metrics_config, X_test=None, **kwargs):
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
    res = evaluator.evaluate(y_test, y_pred, y_pred_proba=y_pred_proba, X_test=X_test, **kwargs)
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


def validate_and_transform_multilabel(X_test, y_test, y_pred=None, y_transformer=None, class_names=None):
    """Validate multilabel data.

    Args:
        X_test (_type_): _description_
        y_test (_type_): _description_
        y_pred (_type_, optional): _description_. Defaults to None.
        y_transformer (_type_, optional): _description_. Defaults to None.
        class_names (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if class_names:
        labels = class_names
    else:
        labels = set()
        for row in y_test.values:
            for lab in row:
                if lab == "":
                    continue
                labels.add(lab)
    if not y_transformer:
        logger.warn("No y_transformer found. Using scikit-learn's MultiLabelBinarizer")
        from sklearn.preprocessing import MultiLabelBinarizer
        y_transformer = MultiLabelBinarizer()
        y_transformer.fit(list(labels))
    y_test_ = pd.DataFrame(y_transformer.transform(y_test), columns=y_transformer.classes_, index=X_test.index)

    return y_test_, y_transformer


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


def setup_model_dependencies(requirements):
    """Install pip dependencies dynamically.

    Args:
        requirements (_type_): _description_
    """
    import pip
    logger.info("Installing model dependencies from requirements. %s", requirements)
    pip_args = ["install", "-r", requirements]
    pip.main(pip_args)


def check_and_return_if_mltable(data):
    """Is current director MLTable or not.

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    is_mltable = False
    if os.path.isdir(data):
        local_yaml_path = os.path.join(data, 'MLTable')
        if os.path.exists(local_yaml_path):
            is_mltable = True
    return is_mltable


def read_data(file_path, is_mltable=True, batch_size=None):
    """Util function for reading test data.

    Args:
        file_path (_type_): _description_
        is_mltable: (_type_, optional): _description_. Defaults to True.
        batch_size (_type_): _description_

    Raises:
        DataValidationException: _description_

    Returns:
        _type_: _description_
        _type_: _description_
    """
    if not is_mltable and os.path.isdir(file_path):
        logger.warn("Received URI_FOLDER instead of URI_FILE. Checking if part of LLM Pipeline")
        if constants.LLM_FT_PREPROCESS_FILENAME not in os.listdir(file_path):
            message = "Test Data is a folder. JSON Lines File expected"
            exception = DataLoaderException._with_error(
                AzureMLError.create(BadInputData, error=message)
            )
            log_traceback(exception, logger)
        logger.info("Found LLM Preprocess args")
        import json
        with open(os.path.join(file_path, constants.LLM_FT_PREPROCESS_FILENAME)) as f:
            llm_preprocess_args = json.load(f)
        test_data_path = llm_preprocess_args[constants.LLM_FT_TEST_DATA_KEY]
        file_path = os.path.join(file_path, test_data_path)
    try:
        if is_mltable:
            data = iter([load(file_path).to_pandas_dataframe()])
        else:
            data = read_dataframe(file_path, batch_size=batch_size)
            if not batch_size:
                data = iter([data])
    except Exception as e:
        exception = DataLoaderException._with_error(
            AzureMLError.create(BadInputData, error=repr(e))
        )
        exception.inner_exception = e
        log_traceback(exception, logger)
        raise exception
    return data


def read_dataframe(file_path, batch_size=None):
    """Util function for reading a DataFrame based on the file extension.

    Args:
        file_path (_type_): _description_
        batch_size: (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == '.csv':
        # Reading a CSV file with the specified batch_size
        return pd.read_csv(file_path, chunksize=batch_size)
    elif file_extension == '.tsv':
        # Reading a TSV file with the specified batch_size and skipping initial spaces
        return pd.read_csv(file_path, sep='\t', chunksize=batch_size, skipinitialspace=True)
    elif file_extension == '.jsonl':
        try:
            # Reading a JSONL file with the specified batch_size
            return pd.read_json(file_path, lines=True, dtype=False, chunksize=batch_size)
        except Exception as e:
            logger.info("Loading the data without chunksize. Exception: {}".format(str(e)))
            return pd.read_json(file_path)
    else:
        # Default to reading JSONL files without raising an exception
        return pd.read_json(file_path, lines=True, dtype=False, chunksize=batch_size)


def prepare_data(data, task, label_column_name=None, _has_multiple_output=False, extra_y_test_cols=None):
    """Prepare data.

    Args:
        data (_type_): _description_
        task (_type_): _description_
        label_column_name (_type_, optional): _description_. Defaults to None.
        input_column_names (_type_, optional): _description_. Defaults to None.
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
        X_test, y_test = X_test.drop(extra_y_test_cols + [label_column_name], axis=1), \
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
                exception = DataLoaderException._with_error(
                    AzureMLError.create(BadRegressionData, error=repr(e)),
                    inner_exception=e
                )
                log_traceback(exception, logger)
                raise exception
    if task == constants.TASK.NER:
        if len(X_test.columns) > 1 and "tokens" not in X_test.columns:
            message = "Too many feature columns in dataset. Only 1 feature column should be passed for NER."
            exception = DataLoaderException(message)
            log_traceback(exception, logger, message)
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
            # print(y_test)
        elif isinstance(y_test.iloc[0], list) or isinstance(y_test.iloc[0], np.ndarray):
            y_test = y_test.apply(lambda x: x[0])
        if not isinstance(y_test.iloc[0], str):
            message = "Ground Truths for Question-answering should be a string \
                       or an array found " + type(y_test.iloc[0])
            exception = DataLoaderException(exception_message=message)
            log_traceback(exception, logger, message)
            raise exception
    if task == constants.TASK.FILL_MASK and y_test is not None:
        if isinstance(y_test.iloc[0], np.ndarray) or isinstance(y_test.iloc[0], list):
            y_test = y_test.apply(lambda x: tuple(x))
        if not isinstance(y_test.iloc[0], str) and not isinstance(y_test.iloc[0], tuple):
            message = "Ground Truths for Fill-Mask should be a string or an array found " + type(y_test.iloc[0])
            exception = DataLoaderException(exception_message=message)
            log_traceback(exception, logger, message)
            raise exception

    if y_test is not None:
        y_test = y_test.values

    if task == constants.TASK.CHAT_COMPLETION:
        X_test = []
        y_test = None
        for conversation in data[label_column_name]:
            df = pd.DataFrame({"input_string": conversation})
            X_test.append(df)

    return X_test, y_test


def fetch_compute_metrics_args(data, task_type):
    """Fetch compute metrics arguments from evaluation config.

    Args:
        data (_type_): _description_
    """
    metrics_args = ArgumentsSet(task_type=task_type)
    metrics_config = {}
    for arg, func in metrics_args.args_set.items():
        val = data.get(arg, None)
        if val is not None:
            try:
                metrics_config[arg] = eval(func)
            except TypeError:
                message = "Invalid dtype passed for config param '" + arg + "'."
                exception = DataValidationException(message)
                log_traceback(exception, logger)
                raise exception

    return metrics_config


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
        import json
        with open(conf_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        exception = DataLoaderException._with_error(
            AzureMLError.create(BadEvaluationConfigFile, error=repr(e)),
            inner_exception=e
        )
        log_traceback(exception, logger, is_critical=True)
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
        import json
        data = json.loads(conf_str)
    except Exception as e:
        message = "Unable to load evaluation_config_params. String is not JSON serialized."
        exception = DataLoaderException._with_error(
            AzureMLError.create(BadEvaluationConfigParam, error=repr(e)),
            inner_exception=e
        )
        log_traceback(exception=exception, logger=logger, message=message, is_critical=True)
        raise exception

    return data


def read_conll(stream_info, labels=None):
    """Read NER data in Conll format.

    Args:
        stream_info (_type_): _description_
        labels (_type_, optional): _description_. Defaults to None.

    Raises:
        DataLoaderException: _description_

    Returns:
        _type_: _description_
    """
    logger.info(type(stream_info))
    if isinstance(stream_info, str):
        with open(stream_info, "r") as f:
            data = f.read()
    elif hasattr(stream_info, "open"):
        f = stream_info.open()
        data = str(f.read())
        f.close()
    else:
        info_link = "https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-\
            standalone-jobs/cli-automl-text-ner-conll/validation-mltable-folder/MLTable"
        raise DataLoaderException("Invalid MLTABLE File for ConLL formatted data. \
                                  See Sample Here : " + info_link)
    data = data.replace("-DOCSTART- O\n\n", "")
    data = data.split("\n\n")

    labels_list = labels
    if labels is None:
        labels_list = []
    tokens, targets = [], []
    for sentence in data:
        toks = sentence.split("\n")
        cur_sentence, cur_target = [], []
        for splits in toks:
            item = splits.split(" ")
            cur_sentence.append(item[0])
            lab = item[-1].strip()
            if lab.isnumeric():
                item[-1] = int(lab)
            else:
                if lab not in labels_list:
                    labels_list.append(lab)
                item[-1] = labels_list.index(lab)

            cur_target.append(item[-1])
        tokens.append(np.array(cur_sentence))
        targets.append(np.array(cur_target))
    return np.asarray(tokens), np.asarray(targets), labels_list


def parse_input_ground_truth_col(col_name):
    """Parse input ground truth columns."""
    extra_cols = None
    if col_name is not None and "," in col_name:
        col_name, extra_cols = col_name.split(",", 1)
        # Adding this to be consistent with how it is being used elsewhere, ideally it should be ""
        col_name = None if col_name == "" else col_name
        extra_cols = extra_cols.split(",") if extra_cols != "" else None
    return col_name, extra_cols
