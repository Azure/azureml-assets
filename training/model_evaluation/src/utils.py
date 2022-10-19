import os
from re import L
import traceback
import numpy as np
import pandas as pd
import json
import pickle
import logging
import azureml.evaluate.mlflow as aml_mlflow

from typing import Tuple, Union
from azureml.core import Run
from azureml.evaluate.mlflow.models.evaluation import EvaluationResult
from constants import TASK, ExceptionTypes
from run_utils import TestRun
from logging_utilities import get_logger, log_traceback
from azureml.metrics.azureml_classification_metrics import AzureMLClassificationMetrics
from azureml.metrics.azureml_regression_metrics import AzureMLRegressionMetrics
from azureml.metrics.azureml_text_ner_metrics import AzureMLTextNERMetrics
from azureml.metrics import _scoring_utilities, constants as metrics_constants
from azureml.telemetry.activity import log_activity
from mlflow.models.evaluation.artifacts import JsonEvaluationArtifact
from mltable import load
#FUTURE IMPORTS
#Forecasting Metrics

from exceptions import DataValidationException, ModelEvaluationException, DataLoaderException

logger = get_logger(name=__name__)




def assert_and_raise(condition, exception_cls, message):
    if not condition:
        exception = exception_cls(message)
        log_traceback(exception, logger, message, is_critical=True)
        raise exception

def load_transformer(filename):
    try:
        with open(filename, "rb") as f:       
            y_transformer = pickle.load(f)
    except Exception as e:
        error_message = "Not able to load y_transformer. Check file format"
        log_traceback(e, logger, error_message)
        raise DataValidationException(error_message)
    return y_transformer 

class Evaluator:
    def __init__(self, task_type, metrics_config):
        if task_type == TASK.CLASSIFICATION:
            self.evaluator = AzureMLClassificationMetrics(**metrics_config)
        if task_type == TASK.REGRESSION:
            self.evaluator = AzureMLRegressionMetrics(**metrics_config)
        if task_type == TASK.NER:
            self.evaluator = AzureMLTextNERMetrics(**metrics_config)
        
    def evaluate(self, y_test, y_pred):
        return self.evaluator.compute(y_test=y_test, y_pred=y_pred)

def _log_metrics(metrics, artifacts):
    table_scores = {}
    nonscalar_scores = {}
    run = TestRun().run
    
    for name, score in artifacts.items():
        if score is None:
            continue
        elif _scoring_utilities.is_table_metric(name):
            table_scores[name] = score
        elif name in metrics_constants.Metric.NONSCALAR_FULL_SET:
            nonscalar_scores[name] = score
        elif name in metrics_constants.TrainingResultsType.ALL_TIME:
            # Filter out time metrics as we do not log these
            pass
        else:
            logger.warning("Unknown metric {}. Will not log.".format(name))

    # Log the scalar metrics. (Currently, these are stored in CosmosDB)
    for name, score in metrics.items():
        try:
            run.log(name, score)
        except Exception:
            raise ModelEvaluationException(f"Failed to log scalar metric {name} with value {score}")

    for name, score in table_scores.items():
        try:
            run.log_table(name, score)
        except Exception:
            raise ModelEvaluationException(f"Failed to log table metric {name} with value {score}")

    # Log the non-scalar metrics. (Currently, these are all artifact-based.)
    for name, score in nonscalar_scores.items():
        try:
            if name == metrics_constants.Metric.AccuracyTable:
                run.log_accuracy_table(name, score)
            elif name == metrics_constants.Metric.ConfusionMatrix:
                run.log_confusion_matrix(name, score)
            elif name == metrics_constants.Metric.Residuals:
                run.log_residuals(name, score)
            elif name == metrics_constants.Metric.PredictedTrue:
                run.log_predictions(name, score)
            elif name in metrics_constants.Metric.NONSCALAR_FORECAST_SET:
                # Filter out non-scalar forecasting metrics as we do not log these yet
                pass
            else:
                logger.warning("Unsupported non-scalar metric {}. Will not log.".format(name))
        except Exception:
            raise ModelEvaluationException(f"Failed to log non-scalar metric {name} with value {score}")
            
def evaluate_predictions(y_test, y_pred, task_type, metrics_config):
    evaluator = Evaluator(task_type, metrics_config)
    res = evaluator.evaluate(y_test, y_pred)
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
    

class ArgumentsSet:
    def __init__(self, task_type) -> None:
        if task_type == TASK.CLASSIFICATION:
            self.args_set = self.classification
        if task_type == TASK.REGRESSION:
            self.args_set = self.regression
        if task_type == TASK.NER:
            self.args_set = self.text_ner
        
    @property
    def classification(self):
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
        args_map = {
            "metrics": "list(val)",
            "train_label_list": "list(val)"
        }
        return args_map

def validate_and_transform_multilabel(X_test, y_test, y_pred=None, y_transformer=None, class_names=None):
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
    """y_test_ = y_test.apply(lambda x: y_transformer.transform([x])[0])
    y_pred_ = None
    if y_pred:
        y_pred_ = y_pred.apply(lambda x: y_transformer.transform([x])[0])"""
    y_test_ = pd.DataFrame(y_transformer.transform(y_test), columns=y_transformer.classes_, index=X_test.index)
    y_pred_ = None
    if y_pred:
        y_pred_ = y_pred.apply(lambda x: y_transformer.transform([x])[0])
    
    return y_test_, y_pred_, y_transformer

def setup_model_dependencies(requirements):
    import pip
    logger.info("Installing model dependencies from requirements. %s", requirements)
    pip_args = ["install", "-r", requirements]
    pip.main(pip_args)

def read_data(file_path, label_column_name=None, prediction_column_name=None):
    try:
        tbl = load(file_path)
        df = tbl.to_pandas_dataframe()
        #print(df.head())
    except Exception as e:
        traceback.print_exc()
        error_message = f"Failed to open test data with following error {repr(e)}"
        log_traceback(e, logger, error_message, is_critical=True)
        raise DataValidationException(error_message)

    y_test, y_pred = None, None

    if label_column_name is not None:
        error_message = f"No column with name : '{label_column_name}' found in data"
        assert_and_raise(label_column_name in df.columns, DataValidationException, error_message)    
        y_test = df[label_column_name]
        df.drop(label_column_name, axis=1, inplace=True)

    if prediction_column_name is not None:
        error_message = f"No column with with name : '{prediction_column_name}' found in data"
        assert_and_raise(prediction_column_name in df.columns, DataValidationException, error_message)    
        y_pred = df[prediction_column_name]
        df.drop(prediction_column_name, axis=1, inplace=True)
    
    return df, y_test, y_pred

def read_config(conf_folder, task_type):
    try:
        json_file = load(conf_folder)
        data = json_file.to_pandas_dataframe().to_dict(orient="records")[0]
    except Exception as e:
        error_message = f"Failed to load config file with error {repr(e)}"
        log_traceback(e, logger, error_message, is_critical=True)
        raise DataValidationException(error_message)
    label_column_name = data.get("label_column_name", None)
    prediction_column_name = data.get("prediction_column_name", None)

    metrics_args = ArgumentsSet(task_type=task_type)
    metrics_config = {}
    #train_label_list = data.get("train_label_list", None)
    #metrics_config["train_label_list"] = train_label_list
    for arg, func in metrics_args.args_set.items():
        val = data.get(arg, None)
        if val is not None:
            if arg == "y_transformer":
                val = os.path.join(conf_folder, val)
            metrics_config[arg] = eval(func)
    
    return label_column_name, prediction_column_name, metrics_config

def _validate_ner_line(line):
    assert_and_raise(condition=(line == '\n' or line.count(' ') == 1), 
                    exception_cls=DataValidationException,
                    message="Invalid or Unsupported NER Data Format")   

    if line != '\n':
        token, label = line.split(' ')
        assert_and_raise(
            condition=len(token) != 0,
            exception_cls=DataValidationException,
            message="Invalid or Unsupported NER Data Format"
        )

        assert_and_raise(
            condition=(label.strip() == "O" or label.startswith("I-") or label.startswith("B-")),
            exception_cls=DataValidationException,
            message="Invalid Label format"
        )

def read_conll(stream_info, labels=None):
    logger.info(type(stream_info))
    if isinstance(stream_info, str):
        with open(stream_info, "r") as f:
            data = f.read()
    elif hasattr(stream_info, "open"):
        f = stream_info.open()
        data = str(f.read())
        f.close()
    else:
        raise DataLoaderException("Invalid MLTABLE File for ConLL formatted data. See Sample Here : https://github.com/Azure/azureml-examples/blob/main/cli/jobs/automl-standalone-jobs/cli-automl-text-ner-conll/validation-mltable-folder/MLTable")
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
            #_validate_ner_line(sentence)
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
