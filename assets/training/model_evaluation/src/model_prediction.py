# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import azureml.evaluate.mlflow as aml_mlflow
import pandas as pd
import constants
import torch
import ast
from itertools import repeat

from exceptions import (
    ModelValidationException,
    DataLoaderException,
    PredictException,
    ModelEvaluationException,
    swallow_all_exceptions
)
from error_definitions import ModelPredictionInternalError, BadModel, BadInputData
from logging_utilities import custom_dimensions, current_run, get_logger, log_traceback
from azureml.telemetry.activity import log_activity
from image_constants import ImageDataFrameParams
from utils import (
    ArgumentParser,
    check_and_return_if_mltable,
    get_predictor,
    read_data,
    prepare_data,
    filter_pipeline_params
)
from validation import _validate, validate_args
from azureml._common._error_definition.azureml_error import AzureMLError

logger = get_logger(name=__name__)
custom_dimensions.app_name = constants.TelemetryConstants.MODEL_PREDICTION_NAME
# current_run = TestRun()
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
custom_dims_dict = vars(custom_dimensions)


class ModelPredictionRunner:
    """Main Class for Inferencing all tasks modes."""

    def __init__(self, model_uri, task, device,
                 batch_size, config):
        """__init__.

        Args:
            model_uri (_type_): _description_
            task (_type_): _description_
            device (_type_): _description_

        Raises:
            ModelValidationException: _description_
        """
        self.model_uri = model_uri
        self.task = task
        self.config = config
        self.multilabel = task in constants.MULTILABEL_SET

        self.device = device
        if device == constants.DEVICE.CPU:
            self.device = -1
        elif device == constants.DEVICE.GPU:
            if torch.cuda.is_available():
                self.device = torch.cuda.current_device()
            else:
                logger.warning("No GPU found. Using CPU instead.")
                self.device = -1
        self.batch_size = batch_size
        self.masks_required = True if task == constants.TASK.IMAGE_INSTANCE_SEGMENTATION else False

        with log_activity(logger, constants.TelemetryConstants.LOAD_MODEL,
                          custom_dimensions=custom_dims_dict):
            logger.info("Loading model.")
            try:
                predictor_cls = get_predictor(self.task)
                self.predictor = predictor_cls(self.model_uri, self.task, self.device)
                logger.info(
                    f"Model loaded, Device: {getattr(self.predictor.model, 'device', 'not present')}")
            except Exception as e:
                exception = ModelValidationException._with_error(
                    AzureMLError.create(BadModel, error=repr(e)),
                    inner_exception=e
                )
                log_traceback(exception, logger)
                raise exception

    def load_data(self, test_data, label_column_name=None, input_column_names=None, is_mltable=True):
        """Load Data.

        Args:
            test_data (_type_): _description_
            label_column_name (_type_, optional): _description_. Defaults to None.
            input_column_names (list): Name of input column names

        Raises:
            DataLoaderException: _description_

        Returns:
            _type_: _description_
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
        data = map(prepare_data, data, repeat(self.task), repeat(label_column_name))
        return data  # X_test, y_test

    def predict(self, data, label_column_name):
        """Predict.

        Args:
            data (_type_): _description_
            label_column_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        predictions, pred_probas, y_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        for idx, (X_test, y_test_chunk) in enumerate(data):
            logger.info("batch: " + str(idx))
            y_transformer = None

            pred_probas_chunk = None
            pipeline_params = filter_pipeline_params(self.config)
            torch_error_message = "Model prediction Failed.\nPossible Reason:\n" \
                                  "1. Your input text exceeds max length of model.\n" \
                                  "\t\tYou can either keep truncation=True in tokenizer while logging model.\n" \
                                  "\t\tOr you can pass tokenizer_config in evaluation_config.\n" \
                                  "2. Your tokenizer's vocab size doesn't match with model's vocab size.\n" \
                                  "\t\tTo fix this check your model/tokenizer config.\n" \
                                  "3. If it is Cuda Assertion Error, check your test data." \
                                  "Whether that input can be passed directly to model or not."
            try:
                if self.task == constants.TASK.TRANSLATION:
                    source_lang = self.config.get("source_lang", None)
                    target_lang = self.config.get("target_lang", None)
                    predictions_chunk = self.predictor.predict(X_test, y_transformer=y_transformer,
                                                               multilabel=self.multilabel,
                                                               source_lang=source_lang, target_lang=target_lang)
                else:
                    # batching is handled in mlflow predict for image tasks.
                    if self.task in constants.IMAGE_TASKS:
                        pipeline_params.update(self.config)
                        if self.batch_size:
                            pipeline_params.update({"batch_size": self.batch_size})
                    predictions_chunk = self.predictor.predict(X_test, y_transformer=y_transformer,
                                                               multilabel=self.multilabel,
                                                               masks_required=self.masks_required,
                                                               **pipeline_params)
                if self.task in constants.CLASSIFICATION_SET:
                    pred_probas_chunk = self.predictor.predict_proba(X_test, y_transformer=y_transformer,
                                                                     multilabel=self.multilabel,
                                                                     **pipeline_params)

                if not isinstance(predictions_chunk, pd.DataFrame):
                    predictions_df = pd.DataFrame()
                    predictions_df["predictions"] = predictions_chunk
                    predictions_chunk = predictions_df
                if pred_probas_chunk is not None and not isinstance(pred_probas_chunk, pd.DataFrame):
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
                if self.task in [constants.TASK.IMAGE_OBJECT_DETECTION, constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
                    y_test["image_meta_info"] = X_test["image_meta_info"]

            except Exception as e:
                if isinstance(e, ModelEvaluationException):
                    exception = e
                else:
                    exception = PredictException._with_error(
                        AzureMLError.create(ModelPredictionInternalError, error=repr(e)),
                        inner_exception=e
                    )
                if type(e) in [IndexError, RuntimeError]:
                    log_traceback(exception, logger, torch_error_message)
                else:
                    log_traceback(exception, logger)
                raise exception

        return predictions, pred_probas, y_test


@swallow_all_exceptions(logger)
def run():
    """Entry function of model_test script."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest="task", required=True, choices=constants.ALL_TASKS)
    parser.add_argument("--data", type=str, dest="data", required=True)
    parser.add_argument("--mlflow-model", type=str, dest="mlflow_model", required=True)
    parser.add_argument("--label-column-name", type=str, dest="label_column_name", required=False, default=None)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest="input_column_names", required=False, default=None)
    parser.add_argument("--config-file-name", type=str, dest="config_file_name", required=False, default=None)
    parser.add_argument("--config_str", type=str, dest="config_str", required=False, default=None)

    parser.add_argument("--model-uri", type=str, dest="model_uri", required=False, default="")
    parser.add_argument("--device", type=str, dest="device", required=True, choices=constants.ALL_DEVICES,
                        default=constants.DEVICE.AUTO)
    parser.add_argument("--batch-size", type=int, dest="batch_size", required=False, default=None)

    # Outputs
    parser.add_argument("--predictions", type=str, dest="predictions")
    parser.add_argument("--prediction-probabilities", type=str, dest="prediction_probabilities",
                        required=False, default=None)
    parser.add_argument("--ground-truth", type=str, dest="ground_truth", required=False, default=None)

    args, unknown_args_ = parser.parse_known_args()

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args.__dict__))
        validate_args(args)

        model_uri = args.model_uri.strip()
        mlflow_model = args.mlflow_model
        if mlflow_model:
            model_uri = mlflow_model

    runner = ModelPredictionRunner(
        task=args.task,
        model_uri=model_uri,
        device=args.device,
        batch_size=args.batch_size,
        config=args.config
    )

    with log_activity(logger, constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        logger.info("Loading Data.")
        try:
            is_mltable = check_and_return_if_mltable(args.data)
            data = runner.load_data(args.data, args.label_column_name, args.input_column_names, is_mltable)
        except Exception as e:
            exception = DataLoaderException._with_error(
                AzureMLError.create(BadInputData, error=repr(e)),
                inner_exception=e
            )
            log_traceback(exception, logger)
            raise exception

    with log_activity(logger, constants.TelemetryConstants.PREDICT_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Model Prediction.")
        preds, pred_probas, ground_truth = runner.predict(data, args.label_column_name)

    logger.info("Saving outputs.")
    preds.to_json(args.predictions, orient="records", lines=True)
    if pred_probas is not None:
        pred_probas.to_json(args.prediction_probabilities, orient="records", lines=True)
    if ground_truth is not None:
        ground_truth.to_json(args.ground_truth, orient="records", lines=True)
    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ in "__main__":
    run()
