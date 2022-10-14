from distutils.log import error
import os
from unittest.mock import NonCallableMagicMock
from exceptions import ModelEvaluationException, ModelValidationException, DataValidationException, ArgumentValidationException
from constants import TRANSFORMER_KEY, MLTABLE_FILE_NAME
from logging_utilities import get_logger, log_traceback
from utils import read_config, assert_and_raise


logger = get_logger(name=__name__)

def check_model_uri(model_uri):
    if model_uri.startswith("runs:/"):
        return True
    if model_uri.startswith("models:/"):
        return True
    if os.path.exists(model_uri):
        return True
    return False

def _validate_model(args):
    if args.mode != "compute_metrics":
        assert_and_raise(condition=(len(args.model_uri) > 0) or (args.mlflow_model is not None), 
                        exception_cls=ModelValidationException, 
                        message=f"Either Model URI or Mlflow Model is required for mode {args.mode}")

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

def _validate_data(args):
    print(os.listdir(args.data))
    assert_and_raise(
        condition=os.path.exists(os.path.join(args.data, MLTABLE_FILE_NAME)),
        exception_cls=DataValidationException,
        message="Invalid test data file name. Either file is missing or mispelled."
    )

    if args.config_file_name:
        assert_and_raise(
            condition=os.path.exists(args.config_file_name),
            exception_cls=DataValidationException,
            message="Invalid config file name. Either file is missing or mispelled."
        )
        filepath = args.config_file_name
        label_column_name, prediction_column_name, metrics_config = read_config(filepath, args.task)
        if args.mode != "predict":
            assert_and_raise(
                condition=label_column_name is not None,
                exception_cls=DataValidationException,
                message=f"Label Column Name is required for mode {args.mode}. See documentation on how to create a config file."
            )
        
        if args.mode == "compute_metrics":
            assert_and_raise(
                condition=prediction_column_name is not None,
                exception_cls=DataValidationException,
                message=f"Prediction Column Name required for mode {args.mode}. See documentation on how to create a config file."
            )

def _validate_task(args):
    assert_and_raise(
        condition=args.task in ["regression", "forecasting", "classification", "ner"],
        exception_cls=ArgumentValidationException,
        message="Invalid task type. It should be either classification, regression or forecasting"
    )

def _validate_mode(args):
    assert_and_raise(
        condition=args.mode in ["predict", "compute_metrics", "score"],
        exception_cls=ArgumentValidationException,
        message="Invalid mode type. It should be either predict, compute_metrics or score"
    )    

def validate_args(args):
    _validate_task(args)
    _validate_mode(args)
    _validate_model(args)
    _validate_data(args)


def validate_Xy(X_test, y_test, y_pred, mode):
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
