import argparse
import azureml.evaluate.mlflow as aml_mlflow
import numpy as np
import pandas as pd
import os
import constants


from pathlib import Path
from exceptions import ModelEvaluationException, PredictException, ComputeMetricsException, ScoringException, DataLoaderException, ModelValidationException
from logging_utilities import custom_dimensions, get_logger, log_activity, log_traceback
from utils import (read_config, 
                    read_data, 
                    evaluate_predictions,
                    setup_model_dependencies,
                    validate_and_transform_multilabel,
                    read_conll)
from run_utils import TestRun
from validation import validate_args, validate_Xy


logger = get_logger(name=__name__)
current_run = TestRun()
test_run = current_run.run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

class ModelEvaluationRunner:
    def __init__(self,
            test_data: str,
            mode: str,
            task: str,
            output: str,
            custom_dimensions: dict,
            model_uri: str = None,
            config_file: str = None):
        self.test_data = test_data
        self.mode = mode
        self.task = task
        self.output = output
        self.model_uri = model_uri
        self.config_file = config_file
        self.label_column_name, self.prediction_column_name, self.metrics_config = None, None, {}
        if config_file:
            self.label_column_name, self.prediction_column_name, self.metrics_config = read_config(config_file, self.task)
        self._is_multilabel = self.metrics_config.get("multilabel", False) 
        self.custom_dimensions = custom_dimensions
        self._has_multiple_output = self._is_multilabel or self.mode == "token_classification"

    def load_data(self):
        X_test, y_test, y_pred = None, None, None
        conll = False
        if self.mode == "token_classification":
            from mltable import load
            df = load(self.test_data).to_pandas_dataframe()
            if df.shape[0] == 1 and df.shape[1] == 1:    
                labels = self.metrics_config.get("labels_list", [])
                X_test, y_test, labels = read_conll(self.test_data, labels=labels)
                self.metrics_config["labels_list"] = labels
                conll = True
        if not conll:
            X_test, y_test, y_pred = read_data(self.test_data, self.label_column_name, self.prediction_column_name)

        if self.task == "regression":
            import numpy as np
            if y_test is not None:
                y_test = y_test.astype(np.float64)
            if y_pred is not None:
                y_pred = y_pred.astype(np.float64)
        return X_test, y_test, y_pred

    def _set_multilabel_data(self):
        if self._is_multilabel:
            y_transformer = self.metrics_config.get("y_transformer", None)
            self.y_test, self.y_pred, y_transformer = validate_and_transform_multilabel(X_test=self.X_test, y_test=self.y_test, y_pred=self.y_pred, y_transformer=y_transformer)
            self.metrics_config["y_transformer"] = y_transformer
            num_labels = len(y_transformer.classes_)
            self.metrics_config["class_labels"] = np.array([i for i in range(num_labels)], dtype="int32")
            self.metrics_config["train_labels"] = np.array([i for i in range(num_labels)], dtype="int32")

    def _setup_custom_environment(self):
        with log_activity(logger, constants.TelemetryConstants.ENVIRONMENT_SETUP, custom_dimensions=self.custom_dimensions):
            logger.info("Setting up model dependencies")
            try:
                logger.info("Fetching requirements")
                requirements = aml_mlflow.aml.get_model_dependencies(self.model_uri)
            except Exception as e:
                message = f"Failed to fetch requirements from model_uri with error {repr(e)}"
                log_traceback(e, logger, message)
                raise ModelEvaluationException(message, inner_exception=e)
            try:
                logger.info("Installing Dependencies")
                setup_model_dependencies(requirements)
            except Exception as e:
                message = f"Failed to install model dependencies. {repr(e)}"
                log_traceback(e, logger, message=message)
                raise ModelEvaluationException(message, inner_exception=e)
    
    def fetch_mode_runner(self):
        return getattr(self, self.mode)
    
    def predict(self):
        self._setup_custom_environment()
        with log_activity(logger, constants.TelemetryConstants.PREDICT_NAME, custom_dimensions=self.custom_dimensions):
            try:
                model = aml_mlflow.aml.load_model(self.model_uri, constants.MLFLOW_MODEL_TYPE_MAP[self.task])
            except Exception as e:
                message = "Job failed while loading the model"
                log_traceback(e, logger, message)
                raise ModelValidationException(message, inner_exception=e)
            y_pred = model.predict(self.X_test)
            pred_df = self.X_test.copy(deep=True)
            pred_df[self.label_column_name] = self.y_test
            pred_df[constants.PREDICTIONS_COLUMN_NAME] = y_pred
            pred_df.to_csv(Path(self.output) / "predictions.csv", index=False)
            test_run.upload_file(
                name="predictions.csv", path_or_stream=os.path.join(self.output, "predictions.csv"), datastore_name="workspaceblobstore"
            )
        return
    
    def score(self):
        self._setup_custom_environment()
        self._set_multilabel_data()
        with log_activity(logger, constants.TelemetryConstants.MLFLOW_NAME, custom_dimensions=self.custom_dimensions):
            feature_names = self.X_test.columns
            if self._has_multiple_output:
                eval_data = self.X_test.to_numpy()
                targets = self.y_test.to_numpy()
            else:
                eval_data = self.X_test
                eval_data[self.label_column_name] = self.y_test
                targets = self.label_column_name
            self.metrics_config.update(
                {
                    "log_activity": log_activity,
                    #"log_traceback": log_traceback,
                    "custom_dimensions": self.custom_dimensions,
                    "output": self.output
                }
            )
            result = None
            try:
                print(self.metrics_config)
                result = aml_mlflow.evaluate(
                    self.model_uri,
                    eval_data,
                    targets=targets,
                    feature_names=list(feature_names),
                    model_type=constants.MLFLOW_MODEL_TYPE_MAP[self.task],
                    dataset_name=test_run.experiment.name,
                    evaluators=["azureml"],
                    evaluator_config={"azureml":self.metrics_config},
                )
            except Exception as e:
                message = f"mlflow.evaluate failed with {repr(e)}"
                log_traceback(e, logger, message, True)
                raise ScoringException(message, inner_exception=e)
            if result is not None:
                scalar_metrics = result.metrics
                logger.info("Computed metrics:")
                for metrics, value in scalar_metrics.items():
                    formatted = f"{metrics}: {value}"
                    logger.info(formatted)
                    #test_run.log(metrics, value)

        if result:
            result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))
        return
    
    def compute_metrics(self):
        self._set_multilabel_data()
        with log_activity(logger, constants.TelemetryConstants.COMPUTE_METRICS_NAME, custom_dimensions=self.custom_dimensions):
            result = evaluate_predictions(self.y_test, self.y_pred, self.task, self.metrics_config)
            if result:
                result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))
        return

    def run(self):
        with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING, custom_dimensions=self.custom_dimensions):
            try:
                self.X_test, self.y_test, self.y_pred = self.load_data()
            except Exception as e:
                message = "Couldn't load data."
                log_traceback(e, logger, message, True)
                raise DataLoaderException(message, inner_exception=e)
        validate_Xy(self.X_test, self.y_test, self.y_pred, self.mode)
        mode_runner = self.fetch_mode_runner()
        mode_runner()


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=False, default=None)
    parser.add_argument("--task", type=str, dest="task", choices=["classification", "regression", "forecasting", "ner"])
    parser.add_argument("--data", type=str, dest="data")
    parser.add_argument("--mode", type=str, choices=["score", "predict", "compute_metrics"], default="score")
    parser.add_argument("--config-file-name", dest="config_file_name", required=False, type=str, default="")
    parser.add_argument("--output", type=str, dest="output")
    args = parser.parse_args()

    custom_dims_dict = vars(custom_dimensions)
    print(args)
    print(args.data)

    with log_activity(logger, constants.TelemetryConstants.COMPONENT_NAME, custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments")
        with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME, custom_dimensions=custom_dims_dict):
            validate_args(args)

        model_uri = args.model_uri.strip()
        mlflow_model = args.mlflow_model
        if mlflow_model:
            model_uri = mlflow_model
        
        runner = ModelEvaluationRunner(
            test_data=args.data,
            mode=args.mode,
            task=args.task,
            output=args.output,
            custom_dimensions=custom_dims_dict,
            model_uri=model_uri,
            config_file=args.config_file_name
        )
        runner.run()
    test_run.complete()
    return
        
        


if __name__ in "__main__":
    test_model()
