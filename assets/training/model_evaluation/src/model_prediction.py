# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import argparse
import azureml.evaluate.mlflow as aml_mlflow
import pandas as pd
import constants
import torch
import ast
import traceback
import mltable
import json
import os
from itertools import repeat

from exceptions import (ModelEvaluationException,
                        ModelValidationException,
                        DataLoaderException,
                        PredictException)
from logging_utilities import custom_dimensions, get_logger, log_traceback
from azureml.telemetry.activity import log_activity
from image_classification_dataset import get_classification_dataset, ImageDataFrameParams
from utils import (setup_model_dependencies,
                   check_and_return_if_mltable,
                   get_predictor,
                   read_data,
                   read_config,
                   prepare_data,
                   filter_pipeline_params,
                   sanitize_device_and_device_map)
from run_utils import TestRun
from validation import _validate, validate_args


logger = get_logger(name=__name__)
current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


class Inferencer:
    """Main Class for Inferencing all tasks modes."""

    def __init__(self, model_uri, task, custom_dimensions, device,
                 batch_size, config_file=None, metrics_config=None):
        """__init__.

        Args:
            model_uri (_type_): _description_
            task (_type_): _description_
            custom_dimensions (_type_): _description_
            device (_type_): _description_

        Raises:
            ModelValidationException: _description_
        """
        self.model_uri = model_uri
        self.task = task
        self.metrics_config = {}
        if config_file:
            self.metrics_config = read_config(config_file, task, for_prediction=True)
        elif metrics_config:
            self.metrics_config = metrics_config
        self.multilabel = bool(task == constants.TASK.CLASSIFICATION_MULTILABEL
                               or task == constants.TASK.TEXT_CLASSIFICATION_MULTILABEL
                               or task == constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL)
        self.custom_dimensions = custom_dimensions
        try:
            if device == "gpu":
                self.device = torch.cuda.current_device()
            elif device == "cpu":
                self.device = -1
            else:
                self.device = device
        except Exception:
            logger.warning("No GPU found. Using CPU instead")
            self.device = -1
        self.batch_size = batch_size
        # self._setup_custom_environment()
        with log_activity(logger, constants.TelemetryConstants.LOAD_MODEL,
                          custom_dimensions=self.custom_dimensions):
            logger.info("Loading model")
            try:
                self.model = aml_mlflow.aml.load_model(self.model_uri, constants.MLFLOW_MODEL_TYPE_MAP[self.task])
            except Exception as e:
                traceback.print_exc()
                message = "Job failed while loading the model"
                log_traceback(e, logger, message)
                raise ModelValidationException(message, inner_exception=e)

    def _setup_custom_environment(self):
        """Set Custom dimensions.

        Raises:
            ModelEvaluationException: _description_
            ModelEvaluationException: _description_
        """
        with log_activity(logger, constants.TelemetryConstants.ENVIRONMENT_SETUP,
                          custom_dimensions=self.custom_dimensions):
            logger.info("Setting up model dependencies")
            try:
                logger.info("Fetching requirements")
                requirements = aml_mlflow.pyfunc.get_model_dependencies(self.model_uri)
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

    def load_data(self, test_data, label_column_name=None, input_column_names=None, is_mltable=True):
        """Load Data.

        Args:
            test_data (_type_): _description_
            label_column_name (_type_, optional): _description_. Defaults to None.
            input_column_names (list): Name of input column names

        Raises:
            DataLoaderException: _description_
            DataLoaderException: _description_

        Returns:
            _type_: _description_
        """
        if self.task in [
            constants.TASK.IMAGE_CLASSIFICATION,
            constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL
        ]:
            df = get_classification_dataset(testing_mltable=test_data, multi_label=self.multilabel)
            data = iter([df])
            input_column_names = [ImageDataFrameParams.IMAGE_COLUMN_NAME]
            label_column_name = ImageDataFrameParams.LABEL_COLUMN_NAME
        else:
            data = read_data(test_data, is_mltable)
        data = map(_validate, data, repeat(input_column_names), repeat(label_column_name))
        data = map(prepare_data, data, repeat(self.task), repeat(label_column_name))
        return data  # X_test, y_test

    def predict(self, test_data, label_column_name, input_column_names, is_mltable=True):
        """Predict.

        Args:
            test_data (_type_): _description_
            label_column_name (_type_): _description_
            input_column_names (list): Name of input column names

        Returns:
            _type_: _description_
        """
        predictions, pred_probas, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        with log_activity(logger, constants.TelemetryConstants.DATA_LOADING,
                          custom_dimensions=self.custom_dimensions):
            data = self.load_data(test_data, label_column_name, input_column_names, is_mltable=is_mltable)

        for idx, (X_test, y_test_chunk) in enumerate(data):
            logger.info("batch: " + str(idx))
            y_transformer = None
            predictor_cls = get_predictor(self.task)
            predictor = predictor_cls(self.model)
            pred_probas_chunk = None
            pipeline_params = filter_pipeline_params(self.metrics_config)
            pipeline_params, device = sanitize_device_and_device_map(pipeline_params, self.device)
            torch_error_message = "Model prediction Failed.\nPossible Reason:\n"\
                                  "1. Your input text exceeds max length of model.\n"\
                                  "\t\tYou can either keep truncation=True in tokenizer while logging model.\n"\
                                  "\t\tOr you can pass tokenizer_config in evaluation_config.\n"\
                                  "2. Your tokenizer's vocab size doesn't match with model's vocab size.\n"\
                                  "\t\tTo fix this check your model/tokenizer config.\n"\
                                  "3. If it is Cuda Assertion Error, check your test data."\
                                  "Whether that input can be passed directly to model or not."
            try:
                if self.task == constants.TASK.TRANSLATION:
                    source_lang = self.metrics_config.get("source_lang", None)
                    target_lang = self.metrics_config.get("target_lang", None)
                    predictions_chunk = predictor.predict(X_test, device=device,
                                                          y_transformer=y_transformer, multilabel=self.multilabel,
                                                          source_lang=source_lang, target_lang=target_lang)
                else:
                    # batching is handled in mlflow predict for image tasks.
                    if self.task in constants.IMAGE_TASKS:
                        pipeline_params.update(self.metrics_config)
                        if self.batch_size:
                            pipeline_params.update({"batch_size": self.batch_size})
                    predictions_chunk = predictor.predict(X_test, device=device,
                                                          y_transformer=y_transformer,
                                                          multilabel=self.multilabel,
                                                          **pipeline_params)
                if self.task in constants.CLASSIFICATION_SET:
                    pred_probas_chunk = predictor.predict_proba(X_test, device=device,
                                                                y_transformer=y_transformer,
                                                                multilabel=self.multilabel,
                                                                **pipeline_params)
            except IndexError as e:
                raise PredictException(exception_message=torch_error_message,
                                       inner_exception=e)
            except RuntimeError as e:
                raise PredictException(exception_message=torch_error_message,
                                       inner_exception=e)
            if not isinstance(predictions_chunk, pd.DataFrame):
                predictions_df = pd.DataFrame()
                predictions_df["predictions"] = predictions_chunk
                predictions_chunk = predictions_df
            if not isinstance(pred_probas_chunk, pd.DataFrame) and pred_probas_chunk is not None:
                pred_probas_chunk = pd.DataFrame(pred_probas_chunk)
            if y_test_chunk is not None:
                y_test_chunk = pd.DataFrame(y_test_chunk, index=X_test.index, columns=[label_column_name])
                if isinstance(y_test_chunk[label_column_name].iloc[0], str) \
                        and self.task in constants.MULTIPLE_OUTPUTS_SET:
                    y_test_chunk[label_column_name] = y_test_chunk[label_column_name].apply(
                        lambda x: ast.literal_eval(x)
                    )
            else:
                y_test_chunk = pd.DataFrame({})
            predictions_chunk.index = X_test.index

            if pred_probas_chunk is not None:
                pred_probas_chunk.index = X_test.index
            else:
                pred_probas_chunk = pd.DataFrame({})
            predictions = pd.concat([predictions, predictions_chunk], axis=0)
            pred_probas = pd.concat([pred_probas, pred_probas_chunk], axis=0)
            y_test = pd.concat([y_test, y_test_chunk], axis=0)

        return predictions, pred_probas, y_test


def test_model():
    """Entry function of model_test script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=False, default=None)
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS, required=True)
    parser.add_argument("--data", type=str, dest="data", required=False, default=None)
    parser.add_argument("--data-mltable", type=str, dest="data_mltable", required=False, default="")
    parser.add_argument("--label-column-name", type=str, dest="label_column_name", required=False, default=None)
    parser.add_argument("--predictions", type=str, dest="predictions")
    parser.add_argument("--predictions-mltable", type=str, dest="predictions_mltable")
    parser.add_argument("--prediction-probabilities", type=str, required=False,
                        default=None, dest="prediction_probabilities")
    parser.add_argument("--prediction-probabilities-mltable", type=str, required=False,
                        default=None, dest="prediction_probabilities_mltable")
    parser.add_argument("--ground-truth", type=str, required=False, default=None, dest="ground_truth")
    parser.add_argument("--ground-truth-mltable", type=str, required=False, default=None, dest="ground_truth_mltable")
    parser.add_argument("--device", type=str, required=False, default="auto", dest="device")
    parser.add_argument("--batch-size", type=int, required=False, default=None, dest="batch_size")
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest="input_column_names", required=False, default=None)
    parser.add_argument("--config_str", type=str, dest="config_str", required=False, default=None)
    parser.add_argument("--config-file-name", dest="config_file_name", required=False, type=str, default=None)

    args = parser.parse_args()
    # print(args)
    custom_dimensions.app_name = constants.TelemetryConstants.MODEL_PREDICTION_NAME
    custom_dims_dict = vars(custom_dimensions)
    with log_activity(logger, constants.TelemetryConstants.MODEL_PREDICTION_NAME, custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments")
        with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME, custom_dimensions=custom_dims_dict):
            validate_args(args)

        model_uri = args.model_uri.strip()
        mlflow_model = args.mlflow_model
        if mlflow_model:
            model_uri = mlflow_model

        met_config = None
        if args.config_str:
            if args.config_file_name:
                logger.warning("Both evaluation_config and evaluation_config_params are passed. \
                               Using evaluation_config as additional params.")
            else:
                try:
                    met_config = json.loads(args.config_str)
                except Exception as e:
                    message = "Unable to load evaluation_config_params. String is not JSON serielized."
                    raise DataLoaderException(message,
                                              inner_exception=e)

        is_mltable, data = check_and_return_if_mltable(args.data, args.data_mltable)

        runner = Inferencer(
            task=args.task,
            custom_dimensions=custom_dims_dict,
            model_uri=model_uri,
            device=args.device,
            batch_size=args.batch_size,
            config_file=args.config_file_name,
            metrics_config=met_config
        )
        try:
            preds, pred_probas, ground_truth = runner.predict(data, label_column_name=args.label_column_name,
                                                              input_column_names=args.input_column_names,
                                                              is_mltable=is_mltable)
        except Exception as e:
            if isinstance(e, PredictException):
                raise e
            message = "Model Prediction Failed. See inner error."
            raise PredictException(exception_message=message,
                                   inner_exception=e)

        preds.to_json(args.predictions, orient="records", lines=True)

        preds_file_name = args.predictions.split(os.sep)[-1]
        preds_mltable_file_path = args.predictions_mltable + os.sep + preds_file_name
        preds.to_json(preds_mltable_file_path, orient="records", lines=True)
        preds_mltable = mltable.from_json_lines_files(paths=[{'file': preds_mltable_file_path}])
        preds_mltable.save(args.predictions_mltable)
        if pred_probas is not None:
            pred_probas.to_json(args.prediction_probabilities, orient="records", lines=True)

            pred_probas_file_name = args.prediction_probabilities.split(os.sep)[-1]
            pred_probas_mltable_file_path = args.prediction_probabilities_mltable + os.sep + pred_probas_file_name
            pred_probas.to_json(pred_probas_mltable_file_path, orient="records", lines=True)
            pred_probas_mltable = mltable.from_json_lines_files(paths=[{'file': pred_probas_mltable_file_path}])
            pred_probas_mltable.save(args.prediction_probabilities_mltable)
        if ground_truth is not None:
            ground_truth.to_json(args.ground_truth, orient="records", lines=True)

            ground_truth_file_name = args.ground_truth.split(os.sep)[-1]
            ground_truth_mltable_file_path = args.ground_truth_mltable + os.sep + ground_truth_file_name
            ground_truth.to_json(ground_truth_mltable_file_path, orient="records", lines=True)
            ground_truth_mltable = mltable.from_json_lines_files(paths=[{'file': ground_truth_mltable_file_path}])
            ground_truth_mltable.save(args.ground_truth_mltable)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ in "__main__":
    test_model()
