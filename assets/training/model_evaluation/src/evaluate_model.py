# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import argparse

import azureml.evaluate.mlflow as aml_mlflow
import os
import json
import constants
import torch
from itertools import repeat
from copy import deepcopy

from exceptions import (ModelEvaluationException,
                        ScoringException,
                        DataLoaderException)
from error_definitions import (ModelEvaluationInternalError,
                               BadEvaluationConfigParam,
                               BadInputData)
from azureml._common._error_definition.azureml_error import AzureMLError
from image_classification_dataset import get_classification_dataset, ImageDataFrameParams
from logging_utilities import custom_dimensions, get_logger, log_traceback
from azureml.telemetry.activity import log_activity
from utils import (read_config,
                   check_and_return_if_mltable,
                   read_data,
                   prepare_data,
                   setup_model_dependencies,
                   get_predictor,
                   filter_pipeline_params,
                   fetch_compute_metrics_args,
                   sanitize_device_and_device_map)

from run_utils import TestRun
from validation import _validate, validate_args, validate_Xy

logger = get_logger(name=__name__)
current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())


class EvaluateModel:
    """EvaluateModel object."""

    def __init__(self,
                 task: str,
                 model_uri: str,
                 output: str,
                 custom_dimensions: dict,
                 device: str = "auto",
                 config_file: str = None,
                 metrics_config: dict = None,
                 batch_size: int = 1) -> None:
        """
        Evaluate Model Object.

        Args:
            task: str
            model_uri: str
            output: str
            custom_dimensions: str
            device: str
            config_file: str
            metrics_config: dict,
            batch_size: int
        """
        self.task = task
        self.model_uri = model_uri
        self.output = output
        self.multilabel = bool(task == constants.TASK.CLASSIFICATION_MULTILABEL or
                               task == constants.TASK.TEXT_CLASSIFICATION_MULTILABEL or
                               task == constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL)
        self._has_multiple_output = task in constants.MULTIPLE_OUTPUTS_SET
        self.custom_dimensions = custom_dimensions
        self.batch_size = batch_size
        self.metrics_config = {}
        if config_file:
            self.metrics_config = read_config(config_file, task, for_prediction=True)
        elif metrics_config:
            self.metrics_config = metrics_config
        pipeline_params = filter_pipeline_params(self.metrics_config)
        compute_metrics_config = fetch_compute_metrics_args(self.metrics_config, task_type=self.task)
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
        #pipeline_params, self.device = sanitize_device_and_device_map(pipeline_params, self.device)
        self.metrics_config = deepcopy(pipeline_params)
        self.metrics_config.update(compute_metrics_config)
        logger.info("Logging to check metrics config in evaluate_model: "+str(self.metrics_config))

        # self._setup_custom_environment()

    def _validate_schema(self, X_test):
        self.model = aml_mlflow.aml.load_model(self.model_uri, constants.MLFLOW_MODEL_TYPE_MAP[self.task])
        predictor_cls = get_predictor(self.task)
        predictor = predictor_cls(self.model)
        predictor._ensure_base_model_input_schema(X_test=X_test)
        device = None
        if self.device == "auto":
            model_device = None
            if predictor.is_hf:
                hf_device_map = getattr(predictor.model._model_impl.hf_model, "hf_device_map", None)
                if hf_device_map is not None:
                    logger.info("hf_device_map: "+str(hf_device_map))
                    unique_devices = set(hf_device_map.values())
                    logger.info("Unique devices for model: "+str(unique_devices))
                    if len(unique_devices) == 1:
                        model_device = predictor.model._model_impl.hf_model.device
                    #device = next(iter(hf_device_map.values()))
                else:
                    model_device = predictor.model._model_impl.hf_model.device
            elif predictor.is_torch:
                model_device = predictor.model.getattr("device", None)
            logger.info("Model Device: "+ str(model_device))
            if model_device is not None:
                if model_device.type == "cpu":
                    device = -1
                else:
                    device = model_device.index
            logger.info("Device: "+str(device))
        else:
            device = self.device
        self.device = device

    def load_data(self, test_data, label_column_name, input_column_names=None, is_mltable=True):
        """
        Load data in required format.

        Args:
            test_data: jsonlines
            label_column_name: str
            input_column_names: list
            is_mltable: boolean

        Returns: Dataframe

        """
        if self.task in [constants.TASK.IMAGE_CLASSIFICATION,
                         constants.TASK.IMAGE_CLASSIFICATION_MULTILABEL]:
            df = get_classification_dataset(testing_mltable=test_data, multi_label=self.multilabel)
            data = iter([df])
            input_column_names = [ImageDataFrameParams.IMAGE_COLUMN_NAME]
            label_column_name = ImageDataFrameParams.LABEL_COLUMN_NAME
        else:
            data = read_data(test_data, is_mltable)
        data = map(_validate, data, repeat(input_column_names), repeat(label_column_name))
        data = map(prepare_data, data, repeat(self.task), repeat(label_column_name), repeat(self._has_multiple_output))
        return data  # X_test, y_test

    def score(self, test_data, label_column_name, input_column_names, is_mltable=True):
        """
        Evaluate model.

        Args:
            test_data: DataFrame
            label_column_name: str
            input_column_names: list
            is_mltable: boolean

        Returns: None

        """
        with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                          custom_dimensions=self.custom_dimensions) as data_loader_activity:
            try:
                data = self.load_data(test_data, label_column_name, input_column_names, is_mltable=is_mltable)
            except Exception as e:
                message = "Load data failed."
                exception = DataLoaderException._with_error(
                    AzureMLError.create(BadInputData, error=repr(e))
                )
                exception.inner_exception = e
                log_traceback(exception, logger, message, True)
                data_loader_activity.exception(exception.message)
                raise exception

        # No batching support for evaluate model component. Length of data is always 1.
        X_test, y_test = list(data)[0]

        validate_Xy(X_test, y_test)
        self._validate_schema(X_test)

        with log_activity(logger, constants.TelemetryConstants.MLFLOW_NAME, custom_dimensions=self.custom_dimensions):
            feature_names = X_test.columns

            eval_data = X_test
            eval_data[label_column_name] = y_test
            targets = label_column_name
            self.metrics_config.update(
                {
                    "log_activity": log_activity,
                    # "log_traceback": log_traceback,
                    "custom_dimensions": self.custom_dimensions,
                    "output": self.output,
                    "device": self.device,
                    "multi_label": self.multilabel,
                    "batch_size": self.batch_size,
                }
            )
            result = None
            try:
                # print(self.metrics_config)
                result = aml_mlflow.evaluate(
                    self.model,
                    eval_data,
                    targets=targets,
                    feature_names=list(feature_names),
                    model_type=constants.MLFLOW_MODEL_TYPE_MAP[self.task],
                    dataset_name=test_run.experiment.name,
                    evaluators=["azureml"],
                    evaluator_config={"azureml": self.metrics_config},
                )
            except Exception as e:
                message = f"mlflow.evaluate failed with {repr(e)}"
                exception = ScoringException._with_error(
                    AzureMLError.create(ModelEvaluationInternalError, error=repr(e))
                )
                exception.inner_exception = e
                log_traceback(exception, logger, message, True)
                raise exception
            if result is not None:
                scalar_metrics = result.metrics
                logger.info("Computed metrics:")
                for metrics, value in scalar_metrics.items():
                    formatted = f"{metrics}: {value}"
                    logger.info(formatted)

        if result:
            result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))
        return


def test_model():
    """Entry function of model_test script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=False, default=None)
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS, required=True)
    parser.add_argument("--data", type=str, dest="data", required=False, default=None)
    #parser.add_argument("--data-mltable", type=str, dest="data_mltable", required=False, default="")
    parser.add_argument("--config-file-name", dest="config_file_name", required=False, type=str, default=None)
    parser.add_argument("--output", type=str, dest="output")
    parser.add_argument("--device", type=str, required=False, default="auto", dest="device")
    parser.add_argument("--batch-size", type=int, required=False, default=1, dest="batch_size")
    parser.add_argument("--label-column-name", type=str, dest="label_column_name", required=True)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest="input_column_names", required=False, default=None)
    parser.add_argument("--config_str", type=str, dest="config_str", required=False, default=None)
    args = parser.parse_args()
    # logger.info(args)

    custom_dimensions.app_name = constants.TelemetryConstants.EVALUATE_MODEL_NAME
    custom_dims_dict = vars(custom_dimensions)
    # logger.info("Evaluation Config file name:"+args.config_file_name)
    with log_activity(logger, constants.TelemetryConstants.EVALUATE_MODEL_NAME,
                      custom_dimensions=custom_dims_dict) as evaluate_model_activity:
        logger.info("Validating arguments")
        with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                          custom_dimensions=custom_dims_dict) as validation_activity:
            try:
                validate_args(args)
            except Exception as e:
                validation_activity.exception(repr(e))
                raise e

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
                    exception = DataLoaderException._with_error(
                        AzureMLError.create(BadEvaluationConfigParam, error=repr(e))
                    )
                    log_traceback(exception=exception, logger=logger, message=message)
                    evaluate_model_activity.exception(exception.message)
                    raise exception
        data = args.data
        is_mltable = check_and_return_if_mltable(data)
        try:
            runner = EvaluateModel(
                task=args.task,
                output=args.output,
                custom_dimensions=custom_dims_dict,
                model_uri=model_uri,
                config_file=args.config_file_name,
                metrics_config=met_config,
                device=args.device,
                batch_size=args.batch_size
            )
            runner.score(test_data=data, label_column_name=args.label_column_name,
                        input_column_names=args.input_column_names, is_mltable=is_mltable)
        except Exception as e:
            evaluate_model_activity.exception(repr(e))
            raise e

    test_run.add_properties(properties=constants.RUN_PROPERTIES)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ in "__main__":
    test_model()
