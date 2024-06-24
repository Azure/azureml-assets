# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import azureml.evaluate.mlflow as aml_mlflow
import os
import constants
import torch
from itertools import repeat

from constants import ArgumentLiterals

from image_constants import SettingLiterals as ImageSettingLiterals
from exceptions import (
    ScoringException,
    DataLoaderException,
    ModelValidationException,
    DataSavingException,
)
from error_definitions import (
    ModelEvaluationInternalError,
    BadInputData,
    BadModel,
    SavingOutputError,
)
from logging_utilities import (
    custom_dimensions, current_run, get_logger, log_traceback, swallow_all_exceptions, get_azureml_exception
)
from azureml.telemetry.activity import log_activity
from utils import (
    ArgumentParser,
    read_model_prediction_data,
    prepare_data,
    get_predictor,
    filter_pipeline_params,
    fetch_compute_metrics_args
)
from validation import validate_model_prediction_args, validate_common_args, validate_Xy
from task_factory.base import BasePredictor

custom_dimensions.app_name = constants.TelemetryConstants.EVALUATE_MODEL_NAME
logger = get_logger(name=__name__)
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
custom_dims_dict = vars(custom_dimensions)


class EvaluateModel(BasePredictor):
    """EvaluateModel object."""

    def __init__(self,
                 task: str,
                 model_uri: str,
                 output: str,
                 device: str = constants.DEVICE.AUTO,
                 config: dict = None,
                 batch_size: int = None) -> None:
        """
        Evaluate Model Object.

        Args:
            task: str
            model_uri: str
            output: str
            device: str
            config: dict,
            batch_size: int
        """
        super().__init__(model_uri, task, device)

        self.task = task
        self.model_uri = model_uri
        self.output = output
        self.multilabel = task in constants.MULTILABEL_SET
        self._has_multiple_output = task in constants.MULTIPLE_OUTPUTS_SET
        self.batch_size = batch_size
        self.masks_required = True if task == constants.TASK.IMAGE_INSTANCE_SEGMENTATION else False
        self.config = config

        self.current_device = device
        self.device = device
        if device == constants.DEVICE.CPU:
            self.device = -1
        elif device == constants.DEVICE.GPU:
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
            else:
                logger.warning("No GPU found. Using CPU instead.")
                self.device = -1
        logger.info("Logging to check metrics config in evaluate_model: " + str(self.config))

    def _validate_schema(self, X_test):
        predict_fn = self.model.predict
        if self.task == constants.TASK.FORECASTING:
            predict_fn = self.model.forecast
        with log_activity(logger, constants.TelemetryConstants.LOAD_MODEL,
                          custom_dimensions=custom_dims_dict):
            try:
                self.handle_device_failure(X_test, predict_fn=predict_fn)  # Handling device failure
                predictor_cls = get_predictor(self.task)
                predictor = predictor_cls(self.model_uri, self.task, self.device)
                logger.info(f"model loaded, Device: {getattr(predictor.model, 'device', 'not present')}")
            except Exception as e:
                exception = get_azureml_exception(ModelValidationException, BadModel, e, error=repr(e))
                log_traceback(exception, logger)
                raise exception

    def load_data(self, test_data, label_column_name, input_column_names=None):
        """
        Load data in required format.

        Args:
            test_data: jsonlines
            label_column_name: str
            input_column_names: list
            is_mltable: boolean

        Returns: Dataframe

        """
        all_cols = list(input_column_names)
        if label_column_name is not None:
            all_cols += [label_column_name]
        data, _ = read_model_prediction_data(
            test_data, input_column_names, label_column_name, self.task, self.batch_size
        )
        data = map(prepare_data, data, repeat(self.task), repeat(all_cols), repeat(label_column_name),
                   repeat(False), repeat(list()), repeat(self.batch_size))
        return data  # X_test, y_test

    def score(self, data, label_column_name):
        """
        Evaluate model.

        Args:
            data: DataFrame
            label_column_name: str
            input_column_names: list
            is_mltable: boolean

        Returns: None

        """
        # No batching support for evaluate model component. Length of data is always 1.
        X_test, y_test = list(data)[0]

        validate_Xy(X_test, y_test)
        self._validate_schema(X_test)

        with log_activity(logger, constants.TelemetryConstants.MLFLOW_NAME, custom_dimensions=custom_dims_dict):
            feature_names = X_test.columns

            eval_data = X_test
            eval_data[label_column_name] = y_test
            targets = label_column_name
            self.config.update(
                {
                    "log_activity": log_activity,
                    "custom_dimensions": custom_dims_dict,
                    "output": self.output,
                    "device": self.device,
                    "multi_label": self.multilabel,
                    "batch_size": self.batch_size,
                    ImageSettingLiterals.MASKS_REQUIRED: self.masks_required,
                    # Image ML classification, identifies task as "multilabel" in azureml-evaluate-mlflow package
                    "multilabel": self.multilabel,
                }
            )
            result = None
            try:
                try:
                    dataset_name = test_run.experiment.name
                except Exception:
                    dataset_name = "test_run.experiment.name"

                result = aml_mlflow.evaluate(
                    self.model,
                    eval_data,
                    targets=targets,
                    feature_names=list(feature_names),
                    model_type=constants.MLFLOW_MODEL_TYPE_MAP[self.task],
                    dataset_name=dataset_name,
                    evaluators=["azureml"],
                    evaluator_config={"azureml": self.config},
                )
            except RuntimeError:
                self.handle_device_failure(X_test)

            except Exception as e:
                exception = get_azureml_exception(ScoringException, ModelEvaluationInternalError, e, error=repr(e))
                log_traceback(exception, logger)
                raise exception
        return result

    def log_and_write_outputs(self, result):
        """Log and Save Outputs."""
        if result is not None:
            scalar_metrics = result.metrics
            logger.info("Computed metrics:")
            for metrics, value in scalar_metrics.items():
                formatted = f"{metrics}: {value}"
                logger.info(formatted)

            result.save(os.path.join(self.output, constants.EVALUATION_RESULTS_PATH))


@swallow_all_exceptions(logger)
def run():
    """Entry function of evaluate model script."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest=ArgumentLiterals.TASK, required=True, choices=constants.ALL_TASKS)
    parser.add_argument("--data", type=str, dest=ArgumentLiterals.DATA, required=True, default=None)
    parser.add_argument("--config-file-name", type=str, dest=ArgumentLiterals.CONFIG_FILE_NAME,
                        required=False, default=None)
    parser.add_argument("--label-column-name", type=str, dest=ArgumentLiterals.LABEL_COLUMN_NAME, required=True)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest=ArgumentLiterals.INPUT_COLUMN_NAMES, required=False, default=None)
    parser.add_argument("--mlflow-model", type=str, dest=ArgumentLiterals.MLFLOW_MODEL, required=True, default=None)
    parser.add_argument("--device", type=str, dest=ArgumentLiterals.DEVICE, required=True,
                        choices=constants.ALL_DEVICES, default=constants.DEVICE.AUTO)
    parser.add_argument("--batch-size", type=int, dest=ArgumentLiterals.BATCH_SIZE, required=False, default=None)
    parser.add_argument("--config_str", type=str, dest=ArgumentLiterals.CONFIG_STR, required=False, default=None)

    # Outputs
    parser.add_argument("--output", type=str, dest=ArgumentLiterals.OUTPUT)

    args, _ = parser.parse_known_args()
    args = vars(args)

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args))
        validate_common_args(args)
        validate_model_prediction_args(args)

        config = args[ArgumentLiterals.CONFIG]
        pipeline_params = filter_pipeline_params(config)
        config = fetch_compute_metrics_args(config, args[ArgumentLiterals.TASK])
        config.update(pipeline_params)

    data = args[ArgumentLiterals.DATA]

    with log_activity(logger, constants.TelemetryConstants.INITIALISING_RUNNER,
                      custom_dimensions=custom_dims_dict):
        runner = EvaluateModel(
            task=args[ArgumentLiterals.TASK],
            output=args[ArgumentLiterals.OUTPUT],
            model_uri=args[ArgumentLiterals.MLFLOW_MODEL],
            config=config,
            device=args[ArgumentLiterals.DEVICE],
            batch_size=None  # args[ArgumentLiterals.BATCH_SIZE] # No batch size support for evaluate model
        )

    with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        try:
            data = runner.load_data(test_data=data, label_column_name=args[ArgumentLiterals.LABEL_COLUMN_NAME],
                                    input_column_names=args[ArgumentLiterals.INPUT_COLUMN_NAMES])
        except Exception as e:
            exception = get_azureml_exception(DataLoaderException, BadInputData, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception

    with log_activity(logger, constants.TelemetryConstants.EVALUATE_MODEL_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Model Evaluation.")
        result = runner.score(data=data, label_column_name=args[ArgumentLiterals.LABEL_COLUMN_NAME])

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
