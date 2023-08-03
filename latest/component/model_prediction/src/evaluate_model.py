# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import azureml.evaluate.mlflow as aml_mlflow
import os
import constants
import torch
from mlflow.models import Model
from itertools import repeat

from image_constants import ImageDataFrameParams, SettingLiterals as ImageSettingLiterals
from exceptions import (
    ScoringException,
    DataLoaderException,
    ModelValidationException,
    swallow_all_exceptions
)
from error_definitions import (
    ModelEvaluationInternalError,
    BadInputData,
    BadModel
)
from azureml._common._error_definition.azureml_error import AzureMLError
from logging_utilities import custom_dimensions, current_run, get_logger, log_traceback
from azureml.telemetry.activity import log_activity
from utils import (
    ArgumentParser,
    check_and_return_if_mltable,
    read_data,
    prepare_data,
    get_predictor,
    filter_pipeline_params,
    fetch_compute_metrics_args
)
from validation import _validate, validate_args, validate_Xy

logger = get_logger(name=__name__)
custom_dimensions.app_name = constants.TelemetryConstants.EVALUATE_MODEL_NAME
# current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
custom_dims_dict = vars(custom_dimensions)


class EvaluateModel:
    """EvaluateModel object."""

    def __init__(self,
                 task: str,
                 model_uri: str,
                 output: str,
                 device: str = constants.DEVICE.AUTO,
                 config: dict = None,
                 batch_size: int = 1) -> None:
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
        elif device == constants.DEVICE.GPU or task == constants.TASK.NER:
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
            else:
                logger.warning("No GPU found. Using CPU instead.")
                self.device = -1
        logger.info("Logging to check metrics config in evaluate_model: " + str(self.config))

    def _ensure_model_on_cpu(self):
        """Ensure model is on cpu.

        Args:
            model (_type_): _description_
        """
        if self.is_hf:
            if hasattr(self.model._model_impl, "hf_model"):
                self.model._model_impl.hf_model = self.model._model_impl.hf_model.cpu()
            else:
                logger.warning("hf_model not found in mlflow model")
        elif self.is_torch:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model = self.model.cpu()
            elif hasattr(self.model, "_model_impl") and isinstance(self.model._model_impl, torch.nn.Module):
                self.model._model_impl = self.model._model_impl.cpu()
            else:
                logger.warning("Torch model is not of type nn.Module")

    def handle_device_failure(self):
        """Handle device failure."""
        if self.device == constants.DEVICE.AUTO and torch.cuda.is_available():
            try:
                cuda_current_device = torch.cuda.current_device()
                logger.info("Loading model and prediction with cuda current device ")
                if self.current_device != cuda_current_device:
                    logger.info(
                        f"Current Device: {self.current_device} does not match expected device {cuda_current_device}")
                    # self.model = load_model(self.model_uri, cuda_current_device, self.task_type)
                    self.current_device = cuda_current_device
                # self.device = self.current_device
            except Exception as e:
                logger.info("Failed on GPU with error: " + repr(e))
        if self.device != -1:
            logger.warning("Predict failed on GPU. Falling back to CPU")
            try:
                logger.info("Loading model and prediction with cuda current device. Trying CPU ")
                if self.current_device != -1:
                    self.current_device = -1
                    # self._ensure_model_on_cpu()
                self.device = -1
            except Exception as e:
                logger.info("Failed on CPU with error: " + repr(e))
                raise e
        curr_model = Model.load(self.model_uri).flavors
        aml_args = {
            "model_hf_load_kwargs": curr_model.get("model_hf_load_kwargs", {})
        }
        if self.device == constants.DEVICE.AUTO:
            aml_args["model_hf_load_kwargs"]["device_map"] = constants.DEVICE.AUTO
        else:
            aml_args["model_hf_load_kwargs"]["device_map"] = "eval_na"

        self.model = aml_mlflow.aml.load_model(self.model_uri, constants.MLFLOW_MODEL_TYPE_MAP[self.task], **aml_args)

    def _validate_schema(self, X_test):
        with log_activity(logger, constants.TelemetryConstants.LOAD_MODEL,
                          custom_dimensions=custom_dims_dict):
            try:
                self.handle_device_failure()  # Handling device failure
                predictor_cls = get_predictor(self.task)
                predictor = predictor_cls(self.model_uri, self.task, self.device)
                logger.info(f"model loaded, Device: {getattr(predictor.model, 'device', 'not present')}")
            except Exception as e:
                exception = ModelValidationException._with_error(
                    AzureMLError.create(BadModel, error=repr(e)),
                    inner_exception=e
                )
                log_traceback(exception, logger)
                raise exception

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
        if self.task in constants.IMAGE_TASKS:
            from image_dataset import get_image_dataset
            df = get_image_dataset(task_type=self.task, test_mltable=test_data)
            data = iter([df])
            input_column_names = [ImageDataFrameParams.IMAGE_COLUMN_NAME]
            label_column_name = ImageDataFrameParams.LABEL_COLUMN_NAME
            if self.task in [constants.TASK.IMAGE_OBJECT_DETECTION,
                             constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
                input_column_names.append(ImageDataFrameParams.IMAGE_META_INFO)
        else:
            data = read_data(test_data, is_mltable)
        data = map(_validate, data, repeat(input_column_names), repeat(label_column_name))
        data = map(prepare_data, data, repeat(self.task), repeat(label_column_name), repeat(self._has_multiple_output))
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
                    # "log_traceback": log_traceback,
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
                # print(self.config)
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
                self.handle_device_failure()

            except Exception as e:
                message = f"mlflow.evaluate failed with {repr(e)}"
                exception = ScoringException._with_error(
                    AzureMLError.create(ModelEvaluationInternalError, error=repr(e)),
                    inner_exception=e
                )
                log_traceback(exception, logger, message, True)
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
    """Entry function of model_test script."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest="task", required=True, choices=constants.ALL_TASKS)
    parser.add_argument("--data", type=str, dest="data", required=True, default=None)
    parser.add_argument("--config-file-name", type=str, dest="config_file_name", required=False, default=None)
    parser.add_argument("--label-column-name", type=str, dest="label_column_name", required=True)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest="input_column_names", required=False, default=None)
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=True, default=None)
    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--device", type=str, dest="device", required=True, choices=constants.ALL_DEVICES,
                        default=constants.DEVICE.AUTO)
    parser.add_argument("--batch-size", type=int, dest="batch_size", required=False, default=1)
    parser.add_argument("--config_str", type=str, dest="config_str", required=False, default=None)

    # Outputs
    parser.add_argument("--output", type=str, dest="output")

    args, unknown_args_ = parser.parse_known_args()

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args.__dict__))
        validate_args(args)

        config = args.config
        pipeline_params = filter_pipeline_params(config)
        config = fetch_compute_metrics_args(config, args.task)
        config.update(pipeline_params)

        model_uri = args.model_uri.strip()
        mlflow_model = args.mlflow_model
        if mlflow_model:
            model_uri = mlflow_model

    data = args.data
    is_mltable = check_and_return_if_mltable(data)

    runner = EvaluateModel(
        task=args.task,
        output=args.output,
        model_uri=model_uri,
        config=config,
        device=args.device,
        batch_size=args.batch_size
    )

    with log_activity(logger, activity_name=constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        try:
            data = runner.load_data(test_data=data, label_column_name=args.label_column_name,
                                    input_column_names=args.input_column_names, is_mltable=is_mltable)
        except Exception as e:
            message = "Load data failed."
            exception = DataLoaderException._with_error(
                AzureMLError.create(BadInputData, error=repr(e))
            )
            exception.inner_exception = e
            log_traceback(exception, logger, message, True)
            raise exception

    with log_activity(logger, constants.TelemetryConstants.EVALUATE_MODEL_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Model Evaluation.")
        result = runner.score(data=data, label_column_name=args.label_column_name)

    logger.info("Logging and Saving outputs.")
    runner.log_and_write_outputs(result)

    test_run.add_properties(properties=constants.RUN_PROPERTIES)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ in "__main__":
    run()
