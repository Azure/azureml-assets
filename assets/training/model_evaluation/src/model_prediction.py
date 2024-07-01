# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Evaluation Component."""

import azureml.evaluate.mlflow as aml_mlflow
from azureml.automl.core.shared.logging_utilities import mark_path_as_loggable
import pandas as pd
import constants
import torch
import ast
import time
import os
from datetime import datetime, timezone
from itertools import repeat, chain
from constants import ArgumentLiterals

from exceptions import (
    ModelLoadingException,
    DataLoaderException,
    PredictException,
    DataValidationException,
    DataSavingException,
)
from error_definitions import (
    ModelPredictionInternalError,
    ModelPredictionUserError,
    BadModel,
    BadInputData,
    EmptyInputData,
    SavingOutputError,
)
from logging_utilities import (
    custom_dimensions, current_run, get_logger, log_traceback,
    swallow_all_exceptions, flush_logger, get_azureml_exception
)
from azureml.telemetry.activity import log_activity
from utils import (
    ArgumentParser,
    get_predictor,
    read_model_prediction_data,
    prepare_data,
    filter_pipeline_params,
)
from validation import (
    validate_model_prediction_args,
    validate_common_args,
    validate_and_get_columns,
)

# Mark current path as allowed
mark_path_as_loggable(os.path.dirname(__file__))

custom_dimensions.app_name = constants.TelemetryConstants.MODEL_PREDICTION_NAME
logger = get_logger(name=__name__)
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
aml_mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
custom_dims_dict = vars(custom_dimensions)


class ModelPredictionRunner:
    """Main Class for Inferencing all tasks modes."""

    def __init__(self, model_uri, task, device, batch_size,
                 input_column_names, label_column_name, extra_y_test_cols,
                 config):
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
        self.input_column_names = input_column_names
        self.label_column_name = label_column_name
        self.extra_y_test_cols = extra_y_test_cols
        with log_activity(logger, constants.TelemetryConstants.LOAD_MODEL,
                          custom_dimensions=custom_dims_dict):
            logger.info("Loading model.")
            try:
                predictor_cls = get_predictor(self.task)
                self.predictor = predictor_cls(self.model_uri, self.task, self.device)
                logger.info(
                    f"Model loaded, Device: {getattr(self.predictor.model, 'device', 'not present')}")
            except Exception as e:
                exception = get_azureml_exception(ModelLoadingException, BadModel, e, error=repr(e))
                log_traceback(exception, logger)
                raise exception

    def load_data(self, test_data):
        """Load Data.

        Args:
            test_data (_type_): _description_
            input_column_names (list): Name of input column names

        Raises:
            DataLoaderException: _description_

        Returns:
            _type_: _description_
        """
        all_cols = list(self.input_column_names)
        if self.label_column_name is not None:
            all_cols += [self.label_column_name]
        if self.extra_y_test_cols is not None:
            all_cols += self.extra_y_test_cols

        data, file_ext = read_model_prediction_data(
            test_data, self.input_column_names, self.label_column_name, self.task, self.batch_size
        )
        data = map(prepare_data, data, repeat(self.task), repeat(all_cols), repeat(self.label_column_name),
                   repeat(False), repeat(self.extra_y_test_cols), repeat(self.batch_size), repeat(file_ext))
        return data  # X_test, y_test

    def load_tokenizer(self, token_counts_enabled):
        """Load Tokenizer.

        Args:
            token_counts_enabled (boolean): If token counts are enabled.

        Returns:
            _type_: _description_
        """
        tokenizer = None
        if token_counts_enabled:
            logger.info("Loading Tokenizer")
            if self.task in constants.TEXT_TOKEN_TASKS:
                tokenizer_load_start = time.time()
                try:
                    tokenizer = self.predictor.model._model_impl.tokenizer
                    logger.info(f"Loaded Tokenizer: {tokenizer}")
                except Exception as e:
                    logger.info(f"Error encountered in loading the tokenizer: {e}")
                    logger.warning("Tokenizer loading failed with the error above; no token counts will be returned.")
                logger.info(f"Tokenizer load time ms: {(time.time() - tokenizer_load_start) * 1000}")
            else:
                logger.warning("Token counts not supported for this task type; no token counts will be returned.")
        return tokenizer

    def predict(self, data):
        """Predict.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        predictions, pred_probas, y_test, performance = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        enable_token_counts = self.config.get("token_count_per_sample", False)
        enable_character_counts = self.config.get("char_count_per_sample", False)
        tokenizer = self.load_tokenizer(enable_token_counts)

        pipeline_params = filter_pipeline_params(self.config, self.predictor.model_flavor, self.predictor)

        try:
            for idx, (X_test, y_test_batch) in enumerate(data):
                logger.info("batch: " + str(idx))
                if len(X_test) == 0:
                    logger.info("No samples in batch. Skipping.")
                    continue

                y_transformer = None

                pred_probas_batch = None

                def add_to_predict_params_if_applicable(dict_to_add, predict_params):
                    if self.predictor.model_flavor != constants.MODEL_FLAVOR.TRANSFORMERS:
                        predict_params = {**predict_params, **dict_to_add}
                    return predict_params

                predict_params = add_to_predict_params_if_applicable(
                    {"y_transformer": y_transformer, "multilabel": self.multilabel}, {})
                if self.task == constants.TASK.TRANSLATION:
                    source_lang = self.config.get("source_lang", None)
                    target_lang = self.config.get("target_lang", None)
                    start_ms = time.time() * 1000
                    predict_params = add_to_predict_params_if_applicable(
                        {"source_lang": source_lang, "target_lang": target_lang}, predict_params)
                    predictions_batch = self.predictor.predict(X_test, **predict_params)
                    end_ms = time.time() * 1000
                    latency_ms = end_ms - start_ms
                elif self.task == constants.TASK.CHAT_COMPLETION:
                    start_ms = time.time() * 1000
                    predictions_batch = self.predictor.predict(X_test, self.input_column_names,
                                                               y_transformer=y_transformer,
                                                               multilabel=self.multilabel,
                                                               masks_required=self.masks_required,
                                                               **pipeline_params)
                    end_ms = time.time() * 1000
                    latency_ms = end_ms - start_ms
                else:
                    # batching is handled in mlflow predict for image tasks.
                    if self.task in constants.IMAGE_TASKS:
                        pipeline_params.update(self.config)
                        if self.batch_size:
                            pipeline_params.update({ArgumentLiterals.BATCH_SIZE: self.batch_size})
                    start_ms = time.time() * 1000
                    predict_params = add_to_predict_params_if_applicable(
                        {"masks_required": self.masks_required}, predict_params)

                    predictions_batch = self.predictor.predict(X_test, **{**predict_params, **pipeline_params})
                    end_ms = time.time() * 1000
                    latency_ms = end_ms - start_ms
                if self.task in constants.CLASSIFICATION_SET:
                    pred_probas_batch = self.predictor.predict_proba(X_test, **{**predict_params, **pipeline_params})

                if not isinstance(predictions_batch, pd.DataFrame):
                    if self.task == constants.TASK.CHAT_COMPLETION:
                        predictions_batch = pd.DataFrame(predictions_batch)
                    else:
                        predictions_df = pd.DataFrame()
                        predictions_df[constants.PREDICTIONS_COLUMN_NAME] = predictions_batch
                        predictions_batch = predictions_df
                if pred_probas_batch is not None and not isinstance(pred_probas_batch, pd.DataFrame):
                    pred_probas_batch = pd.DataFrame(pred_probas_batch)
                if y_test_batch is not None and self.task != constants.TASK.CHAT_COMPLETION:
                    cols = []
                    if self.extra_y_test_cols is not None:
                        cols += self.extra_y_test_cols
                    if self.label_column_name is not None:
                        cols += [self.label_column_name]
                    y_test_batch = pd.DataFrame(y_test_batch, index=X_test.index, columns=cols)
                    # Below code won't work with extra_cols
                    if self.label_column_name is not None \
                            and isinstance(y_test_batch[self.label_column_name].iloc[0], str) \
                            and self.task in constants.MULTIPLE_OUTPUTS_SET:
                        y_test_batch[self.label_column_name] = y_test_batch[self.label_column_name].apply(
                            lambda x: ast.literal_eval(x)
                        )
                    if self.task == constants.TASK.QnA:
                        for col in X_test.columns:
                            y_test_batch[col] = X_test[col]
                elif self.task == constants.TASK.CHAT_COMPLETION:
                    logger.info("Empty/NaN ground truths will replaced with Empty string values ('').")
                    if self.label_column_name is not None:
                        y_test_batch = pd.DataFrame(y_test_batch, columns=[self.label_column_name]).fillna("")
                    else:
                        logger.info("No label column found. Trying to parse test data for ground truths.")
                        y_test_batch = pd.DataFrame(X_test[self.input_column_names[0]].apply(
                            lambda x: x[-1]['content'] if x[-1]['role'] == 'assistant' else ""))
                else:
                    y_test_batch = pd.DataFrame({})
                if self.task != constants.TASK.CHAT_COMPLETION:
                    predictions_batch.index = X_test.index

                if pred_probas_batch is not None:
                    pred_probas_batch.index = X_test.index
                else:
                    pred_probas_batch = pd.DataFrame({})

                performance_batch = pd.DataFrame({})
                performance_batch[constants.PerformanceColumns.BATCH_SIZE_COLUMN_NAME] = \
                    [len(predictions_batch) for _ in range(len(predictions_batch))]

                start_time_iso_string = datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat()
                end_time_iso_string = datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat()
                performance_batch[constants.PerformanceColumns.START_TIME_COLUMN_NAME] = \
                    [start_time_iso_string for _ in range(len(predictions_batch))]
                performance_batch[constants.PerformanceColumns.END_TIME_COLUMN_NAME] = \
                    [end_time_iso_string for _ in range(len(predictions_batch))]
                performance_batch[constants.PerformanceColumns.LATENCY_COLUMN_NAME] = \
                    [latency_ms for _ in range(len(predictions_batch))]

                if self.task in constants.TEXT_TOKEN_TASKS and enable_character_counts:
                    char_time_start = time.time()
                    if self.task in constants.TEXT_OUTPUT_TOKEN_TASKS:
                        performance_batch[constants.PerformanceColumns.OUTPUT_CHARACTERS_COLUMN_NAME] = \
                            [len(pred) for pred in predictions_batch[predictions_batch.columns.values[0]]]

                    if self.task == constants.TASK.QnA:
                        performance_batch[constants.PerformanceColumns.INPUT_CHARACTERS_COLUMN_NAME] = \
                            [len(q) + len(a) for q, a in zip(X_test[X_test.columns.values[0]],
                                                             X_test[X_test.columns.values[1]])]
                    else:
                        performance_batch[constants.PerformanceColumns.INPUT_CHARACTERS_COLUMN_NAME] = \
                            [len(inp) for inp in X_test[X_test.columns.values[0]]]
                    logger.info(f"Character count time ms: {(time.time() - char_time_start) * 1000}")
                elif self.task not in constants.TEXT_TOKEN_TASKS and enable_character_counts:
                    logger.warning("Character counts not supported for this task type; "
                                   "no character counts will be returned.")

                if tokenizer is not None:
                    token_time_start = time.time()
                    if self.task in constants.TEXT_OUTPUT_TOKEN_TASKS:
                        curr_predictions = list(predictions_batch[predictions_batch.columns.values[0]])
                        tokenized_predictions = tokenizer(curr_predictions)["input_ids"]
                        pred_num_tokens = [len(tokenized_input) for tokenized_input in tokenized_predictions]
                        performance_batch[constants.PerformanceColumns.OUTPUT_TOKENS_COLUMN_NAME] = pred_num_tokens

                    if self.task == constants.TASK.QnA:
                        tokenized_inputs = tokenizer(
                            list(X_test[X_test.columns.values[0]]), list(X_test[X_test.columns.values[1]])
                        )["input_ids"]
                    elif self.task == constants.TASK.CHAT_COMPLETION:
                        input_data = X_test[X_test.columns.values[0]]
                        tokenized_inputs = [
                            list(chain(*tokenizer([chat['content'] for chat in conversation])["input_ids"]))
                            for conversation in input_data
                        ]
                    else:
                        tokenized_inputs = tokenizer(list(X_test[X_test.columns.values[0]]))["input_ids"]
                    input_num_tokens = [len(tokenized_input) for tokenized_input in tokenized_inputs]
                    performance_batch[constants.PerformanceColumns.INPUT_TOKENS_COLUMN_NAME] = input_num_tokens
                    logger.info(f"Token count time ms: {(time.time() - token_time_start) * 1000}")

                predictions = pd.concat([predictions, predictions_batch], axis=0)
                pred_probas = pd.concat([pred_probas, pred_probas_batch], axis=0)
                y_test = pd.concat([y_test, y_test_batch], axis=0)
                performance = pd.concat([performance, performance_batch], axis=0)
                if self.task in [constants.TASK.IMAGE_OBJECT_DETECTION, constants.TASK.IMAGE_INSTANCE_SEGMENTATION]:
                    y_test["image_meta_info"] = X_test["image_meta_info"]

                logger.info(f"Latency (ms) for this batch: {latency_ms}")
        except ValueError as e:
            exception = get_azureml_exception(PredictException, ModelPredictionUserError, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception
        except Exception as e:
            exception = get_azureml_exception(PredictException, ModelPredictionInternalError, e,
                                              wrap_azureml_ex=False, error=repr(e))
            if isinstance(e, (IndexError, RuntimeError)):
                log_traceback(exception, logger, constants.ErrorStrings.TorchErrorMessage)
            else:
                log_traceback(exception, logger)
            raise exception

        if self.batch_size is not None and len(predictions) == 0:
            exception = get_azureml_exception(DataValidationException, EmptyInputData, None)
            log_traceback(exception, logger)
            raise exception
        return predictions, pred_probas, y_test, performance


@swallow_all_exceptions(logger)
def run():
    """Entry function of model prediction script."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--task", type=str, dest=ArgumentLiterals.TASK, required=True, choices=constants.ALL_TASKS)
    parser.add_argument("--data", type=str, dest=ArgumentLiterals.DATA, required=True)
    parser.add_argument("--mlflow-model", type=str, dest=ArgumentLiterals.MLFLOW_MODEL, required=True)
    parser.add_argument("--label-column-name", type=lambda x: x.split(","),
                        dest=ArgumentLiterals.LABEL_COLUMN_NAME, required=False, default=None)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest=ArgumentLiterals.INPUT_COLUMN_NAMES, required=False, default=None)
    parser.add_argument("--config-file-name", type=str, dest=ArgumentLiterals.CONFIG_FILE_NAME,
                        required=False, default=None)
    parser.add_argument("--config_str", type=str, dest=ArgumentLiterals.CONFIG_STR, required=False, default=None)

    parser.add_argument("--device", type=str, dest=ArgumentLiterals.DEVICE, required=True,
                        choices=constants.ALL_DEVICES, default=constants.DEVICE.AUTO)
    parser.add_argument("--batch-size", type=int, dest=ArgumentLiterals.BATCH_SIZE, required=False, default=None)

    # Outputs
    parser.add_argument("--predictions", type=str, dest=ArgumentLiterals.PREDICTIONS)
    parser.add_argument("--prediction-probabilities", type=str, dest=ArgumentLiterals.PREDICTION_PROBABILITIES,
                        required=False, default=None)
    parser.add_argument("--ground-truth", type=str, dest=ArgumentLiterals.GROUND_TRUTHS, required=False, default=None)
    parser.add_argument("--performance-metadata", type=str, dest=ArgumentLiterals.PERFORMANCE_METADATA,
                        required=False, default=None)

    args, _ = parser.parse_known_args()
    args = vars(args)

    with log_activity(logger, constants.TelemetryConstants.VALIDATION_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Validating arguments: " + repr(args))
        validate_common_args(args)
        validate_model_prediction_args(args)

        input_column_names, label_column_name, extra_y_test_cols = validate_and_get_columns(args)

    with log_activity(logger, constants.TelemetryConstants.INITIALISING_RUNNER,
                      custom_dimensions=custom_dims_dict):
        runner = ModelPredictionRunner(
            task=args[ArgumentLiterals.TASK],
            model_uri=args[ArgumentLiterals.MLFLOW_MODEL],
            device=args[ArgumentLiterals.DEVICE],
            batch_size=args[ArgumentLiterals.BATCH_SIZE],
            input_column_names=input_column_names,
            label_column_name=label_column_name,
            extra_y_test_cols=extra_y_test_cols,
            config=args[ArgumentLiterals.CONFIG]
        )

    with log_activity(logger, constants.TelemetryConstants.DATA_LOADING,
                      custom_dimensions=custom_dims_dict):
        logger.info("Loading Data.")
        flush_logger(logger)
        try:
            data = runner.load_data(args[ArgumentLiterals.DATA])
        except Exception as e:
            exception = get_azureml_exception(DataLoaderException, BadInputData, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception

    with log_activity(logger, constants.TelemetryConstants.PREDICT_NAME,
                      custom_dimensions=custom_dims_dict):
        logger.info("Model Prediction.")
        flush_logger(logger)
        preds, pred_probas, ground_truth, perf_data = runner.predict(data)

    with log_activity(logger, activity_name=constants.TelemetryConstants.LOG_AND_SAVE_OUTPUT,
                      custom_dimensions=custom_dims_dict):
        logger.info("Saving outputs.")
        try:
            preds.to_json(args[ArgumentLiterals.PREDICTIONS], orient="records", lines=True)
            if pred_probas is not None:
                pred_probas.to_json(args[ArgumentLiterals.PREDICTION_PROBABILITIES], orient="records", lines=True)
            if ground_truth is not None:
                ground_truth.to_json(args[ArgumentLiterals.GROUND_TRUTHS], orient="records", lines=True)
            if perf_data is not None:
                perf_data.to_json(args[ArgumentLiterals.PERFORMANCE_METADATA], orient="records", lines=True)
        except Exception as e:
            exception = get_azureml_exception(DataSavingException, SavingOutputError, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception

    try:
        root_run.add_properties(properties=constants.ROOT_RUN_PROPERTIES)
    except Exception:
        logger.info("PipelineType is already a property at Root Pipeline Run.")
    test_run.complete()
    return


if __name__ == "__main__":
    run()
