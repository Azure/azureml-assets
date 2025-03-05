# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run script to infer."""
# flake8: noqa: E702
import sys

# sys.path.append("/src/")


import json
import os
import yaml
import torch
import pandas as pd
import numpy as np
import math
import importlib
import time

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from local_constants import ArgumentLiterals, ModelPath, TEXT_TOKEN_TASKS, PerformanceColumns, FILTER_MODEL_PREDICTION_PARAMS
from local_constants import LLM_FT_PREPROCESS_FILENAME, LLM_FT_CHAT_COMPLETION_KEY, ChatCompletionConstants
from itertools import repeat
from accelerate import PartialState
import torch.distributed as dist
from datetime import datetime, timezone


from data_utils import read_model_prediction_data, prepare_data, prepare_chat_data_from_ft_pipeline
from prepare_data import _clean_and_validate_dataset, validate_and_get_columns
from exceptions import PredictException, DataLoaderException, ModelLoadingException
from error_definitions import ModelPredictionInternalError, BadModel, BadInputData

from llm.optimized.inference.constants import EngineName, TaskType, SupportedTask, ALL_TASKS, VLLMSupportedModels, MIISupportedModels
from llm.optimized.inference.fm_score import FMScore
from logging_utilities import get_logger, get_azureml_exception, log_traceback, swallow_all_exceptions
from llm.optimized.inference.managed_inference import MIRPayload
from llm.optimized.inference.model_utils import build_configs_from_model, get_generator_params

logger = get_logger(name=__name__)
DEVICE_COUNT = torch.cuda.device_count()

distributed_state = PartialState()


class Predictor:
    """Predictor class for distributed inference using container."""

    def __init__(self, engine, task_type, extra_params, num_replicas, label_column_name, tokenizer, extra_y_test_cols=None) -> None:
        """Model Predictor.

        Args:
            engine (str): _description_
            task_type (str): _description_
            extra_params (dict): _description_
            num_replicas (int): _description_
            label_column_name (str): _description_
            tokenizer (Tokenizer): _description_
            extra_y_test_cols (str, optional): _description_. Defaults to None.
        """
        self.engine = engine
        self.task_type = task_type
        self.extra_params = extra_params
        self.num_replicas = num_replicas
        self.label_column_name = label_column_name
        self.tokenizer = tokenizer
        self.extra_y_test_cols = extra_y_test_cols
        self._filter_param()
    
    def _filter_param(self):
        if isinstance(self.extra_params, dict):
            for param in FILTER_MODEL_PREDICTION_PARAMS:
                if self.extra_params.get(param, None):
                    logger.info(f"Filtering param {param} from extra_params")
                    params_dict = self.extra_params.pop(param)
                    self.extra_params.update(params_dict)
        logger.info(f"Extra params dict after filtering: {self.extra_params}")
        
    def postprocess(self, result):
        """Post process computed predictions.

        Args:
            result (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred_df, y_test_df, perf_df, y_pred_proba_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for y_pred, y_test, perf, pred_probas in result:
            logger.info(f"Type here as well: {type(y_test)}")
            y_pred_df = pd.concat([y_pred_df, y_pred], axis=0)
            y_test_df = pd.concat([y_test_df, y_test], axis=0)
            perf_df = pd.concat([perf_df, perf], axis=0)
            y_pred_proba_df = pd.concat([y_pred_proba_df, pred_probas], axis=0)
        ground_truth_columns = [self.label_column_name]
        if self.extra_y_test_cols is not None:
            ground_truth_columns += self.extra_y_test_cols
        y_test_df.columns = ground_truth_columns[:]
        return y_pred_df, y_test_df, perf_df, y_pred_proba_df
    
    def predict(self, data):
        """Predict method for full data.

        Args:
            data (_type_): _description_

        Returns:
            _type_: _description_
        """
        with ThreadPoolExecutor(max_workers=self.num_replicas) as executor:
            result = list(executor.map(
                        self.predict_single,
                        data,
                    ))
        return self.postprocess(result)


    def _make_chat_completion_data(self, input_df, last_chats, col_name):
        appended_data = {col_name:[]}
        input_rows = input_df.values.tolist()
        for ind, datarow in enumerate(input_rows):
            conversation = datarow[0]
            updated_conversation = conversation + [{"role":"assistant", "content":last_chats[ind]}]
            appended_data[col_name].append(updated_conversation)
        return pd.DataFrame(appended_data)


    def predict_single(self, data):
        """Predict single batch.

        Args:
            data (_type_): _description_

        Raises:
            exception: _description_

        Returns:
            _type_: _description_
        """
        X_test, y_test = data
        try:
            input_texts = X_test.values.tolist()
            if isinstance(input_texts[0], list):
                if self.task_type == SupportedTask.CHAT_COMPLETION:
                    input_data = []
                    add_generation_prompt = self.extra_params.pop("add_generation_prompt", True)
                    for itext in input_texts:
                        input_data.append(self.tokenizer.apply_chat_template(itext[0], tokenize=False, add_generation_prompt=add_generation_prompt))
                    input_texts = input_data[:]
                    self.extra_params.update({"return_full_text": False})
                    payload = MIRPayload(input_texts, self.extra_params, TaskType.CONVERSATIONAL, False)
                else:
                    input_texts = [i[0] if len(i) == 1 else [j.strip() for j in i] for i in input_texts]
                    if self.task_type == SupportedTask.TEXT_GENERATION:
                        if "return_full_text" not in self.extra_params:
                            self.extra_params["return_full_text"] = False
                    if self.task_type == SupportedTask.QnA:
                        self.extra_params.update({"truncation":"longest_first"})
                    data = {
                            "input_data": {
                                "input_string": input_texts,
                                "parameters": self.extra_params,
                            }
                    }
                    payload = MIRPayload.from_dict(data)
                    payload.update_params(get_generator_params(payload.params))
                    try: 
                        inference_results = self.engine.run(payload)
                    except:
                        try:
                            logger.info("Failed with longest_first")
                            payload.params["truncation"] = "only_second"
                            inference_results = self.engine.run(payload)
                        except:
                            logger.info("Failed with only first")
                            payload.params["truncation"] = "only_first"
                            inference_results = self.engine.run(payload)
            

            
            logger.info(
                f"Processing new request with parameters: {payload.params}"
            )

            inference_results = None
            if self.task_type == SupportedTask.CHAT_COMPLETION:
                payload.convert_query_to_list()
                start_ms = time.time() * 1000
                inference_results = self.engine.run(payload)
                end_ms = time.time() * 1000
                outputs = [res.response for i, res in enumerate(inference_results)]
                pred_probas = [res.scores for res in inference_results]
            else:
                start_ms = time.time() * 1000
                inference_results = self.engine.run(payload)
                end_ms = time.time() * 1000
                if self.task_type == SupportedTask.TEXT_GENERATION:
                    outputs = []
                    for gt, res in zip(input_texts, inference_results):
                        if gt in res.response:
                            outputs.append(res.response[len(gt):])
                        else:
                            outputs.append(res.response)
                else:
                    outputs = [res.response for i, res in enumerate(inference_results)]
                pred_probas = [res.scores for res in inference_results]
                    


            perf_data = [{
                PerformanceColumns.BATCH_SIZE_COLUMN_NAME: len(input_texts),
                PerformanceColumns.START_TIME_COLUMN_NAME: datetime.fromtimestamp(start_ms / 1000, timezone.utc).isoformat(),
                PerformanceColumns.END_TIME_COLUMN_NAME: datetime.fromtimestamp(end_ms / 1000, timezone.utc).isoformat(),
                PerformanceColumns.LATENCY_COLUMN_NAME: end_ms - start_ms,
                PerformanceColumns.OUTPUT_TOKENS_COLUMN_NAME: len(self.tokenizer(pred)) if self.tokenizer is not None else 0,
                PerformanceColumns.OUTPUT_CHARACTERS_COLUMN_NAME: len(pred) if isinstance(pred, str) else 1,
                PerformanceColumns.INPUT_CHARACTERS_COLUMN_NAME: len(gt) if isinstance(gt, str) else 1,
                PerformanceColumns.INPUT_TOKENS_COLUMN_NAME: len(self.tokenizer(gt)) if self.tokenizer is not None else 0
            } for gt, pred in zip(input_texts, outputs)]
            pred_proba_df = pd.DataFrame(pred_probas, index=X_test.index)
            perf_data = pd.DataFrame(perf_data)

            if self.task_type == SupportedTask.CHAT_COMPLETION or self.task_type == TaskType.CONVERSATIONAL:
                pred_df = self._make_chat_completion_data(X_test.copy(deep=True), outputs,
                                                          col_name=ChatCompletionConstants.OUTPUT_FULL_CONVERSATION)
                pred_df[ChatCompletionConstants.OUTPUT] = outputs
                y_test = pd.DataFrame(y_test, columns=["ground_truth"], index=X_test.index)
                # y_test = self._make_chat_completion_data(X_test.copy(deep=True), y_test, col_name="ground_truth")
                return pred_df, y_test, perf_data, pred_proba_df

            pred_df = pd.DataFrame(outputs, index=X_test.index, columns=["prediction"])
            if isinstance(y_test, pd.Series):
                y_test = y_test.to_frame()
            elif isinstance(y_test, np.ndarray) or isinstance(y_test, list):
                y_test = pd.DataFrame(y_test, index=X_test.index)
            return pred_df, y_test, perf_data, pred_proba_df

        except Exception as e:
            exception = get_azureml_exception(PredictException, ModelPredictionInternalError, e,
                                              wrap_azureml_ex=False, error=repr(e))
            log_traceback(exception, logger)
            raise exception


def _init_cuda_visible_devices():
    import torch

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    if (
            "NVIDIA_VISIBLE_DEVICES" in os.environ
            and os.environ["NVIDIA_VISIBLE_DEVICES"] != "all"
    ):
        # map the gpu ids to integers
        gpu_ids = os.environ["NVIDIA_VISIBLE_DEVICES"].split(",")
        gpu_ids = [str(i) for i in range(len(gpu_ids)) if gpu_ids[i] != "-1"]
    elif torch.cuda.is_available():
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    else:
        # if no GPU is available, don't set anything
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)


def get_model_size(model_path):
    """Estimate size of model.

    Args:
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    size = 0
    # get size
    for path, dirs, files in os.walk(model_path):
        for f in files:
            fp = os.path.join(path, f)
            size += os.path.getsize(fp)
    
    size /= (pow(1024, 3))
    return size


def get_best_engine(config_path, model_path):
    """Fetch best engine for model from architecture.

    Args:
        config_path (_type_): _description_
        model_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(config_path) as f:
        model_config = json.load(f)
    model_class = model_config["architectures"][0]
    folder_size = get_model_size(model_path)
    dtype = model_config.get("torch_dtype", "float32")
    if "float16" in dtype:
        model_size = folder_size//2
    else:
        model_size = folder_size//4
    best_engine = EngineName.HF
    if model_class in VLLMSupportedModels.Models:
        best_engine = EngineName.VLLM
        # TODO: Add logic for selecting MII Over VLLM using model size 
    elif model_class in MIISupportedModels.Models:
        best_engine = EngineName.MII
    return best_engine



def load_data(task, test_data, label_column_name, input_column_names, extra_y_test_cols, batch_size):
    """Load input data.

    Args:
        task (_type_): _description_
        test_data (_type_): _description_
        label_column_name (_type_): _description_
        input_column_names (_type_): _description_
        extra_y_test_cols (_type_): _description_
        batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """
    all_cols = list(input_column_names)
    if label_column_name is not None:
        all_cols += [label_column_name]
    if extra_y_test_cols is not None:
        all_cols += extra_y_test_cols

    data = read_model_prediction_data(file_path=test_data, batch_size=batch_size)
    if task == SupportedTask.CHAT_COMPLETION and os.path.isdir(test_data) and LLM_FT_PREPROCESS_FILENAME in os.listdir(test_data):
        logger.info(f"Run from Finetune Pipeline. Fetching chat completion data from {test_data}")
        data = map(prepare_chat_data_from_ft_pipeline, data)
        return data
    data = map(_clean_and_validate_dataset, data, repeat(all_cols), repeat(batch_size))
    data = map(prepare_data, data, repeat(task), repeat(label_column_name),
                repeat(False), repeat(extra_y_test_cols))
    return data


def _gather_predictions_deprecated(all_preds):
    preds, ground_truth, perf = [], [], []
    for res in all_preds:
        for ind, i in enumerate(res["index"]):
            preds.append((i, res["predictions"][ind]))
            ground_truth.append((i, res["ground_truth"][ind]))
    preds.sort(key=lambda x: x[0])
    ground_truth.sort(key=lambda x: x[0])
    return [j for i, j in preds], [j for i, j in ground_truth]


def _gather_predictions(all_preds):
    preds, ground_truth, perf, pred_probas = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for res in all_preds:
        preds = pd.concat([preds, res["predictions"]], axis=0)
        ground_truth = pd.concat([ground_truth, res["ground_truth"]], axis=0)
        perf = pd.concat([perf, res["perf"]], axis=0)
        pred_probas = pd.concat([pred_probas, res["pred_probas"]], axis=0)
    preds.sort_index(inplace=True)
    ground_truth.sort_index(inplace=True)
    perf.sort_index(inplace=True)
    pred_probas.sort_index(inplace=True)
    return preds, ground_truth, perf, pred_probas


def get_smart_defaults(model_path: str):
    """Compute tensor parallel and num_replicas from model and GPUs available.

    Args:
        model_path (str): _description_

    Raises:
        ValueError

    Returns:
        _type_: Tuple(int, int)
    """
    model_size_in_gb = get_model_size(model_path)
    avg_gpu_free_mem = sum([torch.cuda.mem_get_info(i)[0] for i in range(DEVICE_COUNT)])/DEVICE_COUNT
    avg_gpu_free_mem = avg_gpu_free_mem / pow(1024, 3) # Bytes to GBs
    logger.info(f"Got Model Size {model_size_in_gb} and average GPU memory per Device: {avg_gpu_free_mem}")
    num_possible_replicas = int(DEVICE_COUNT / math.ceil((model_size_in_gb / 0.8) / avg_gpu_free_mem))
    if num_possible_replicas == 0:
        logger.debug(
            "Tensor parallel / model replica calculation with extra memory for cache "
            "results in 0 replicas. Calculating without extra memory for cache.",
        )
        num_possible_replicas = int(DEVICE_COUNT / math.ceil((model_size_in_gb) / avg_gpu_free_mem))
        if num_possible_replicas == 0:
            raise ValueError("Not enough GPU to support model. Please select bigger SKU.")
    tensor_parallel = DEVICE_COUNT//num_possible_replicas
    return tensor_parallel, num_possible_replicas


def load_tokenizer(tokenizer_path, tokenizer_class, **tokenizer_load_kwargs):
    """Load model's tokenizer.

    Args:
        tokenizer_path (str): _description_
        tokenizer_class (str): _description_

    Raises:
        exception: ModelLoadingException

    Returns:
        _type_: Tokenizer
    """
    module_name = "transformers"
    try:
        model_module = importlib.import_module(module_name)
        object_class = getattr(model_module, tokenizer_class)
    except (AttributeError, ImportError) as exc:
        exception = get_azureml_exception(ModelLoadingException, ModelPredictionInternalError, exc,
                                              wrap_azureml_ex=False, error=repr(exc))
        log_traceback(exception, logger)
        raise exception
    tokenizer = object_class.from_pretrained(tokenizer_path, **tokenizer_load_kwargs)
    return tokenizer


def strtobool(val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def is_fsdp_enabled():
    """Torch Fully Sharded Data Parallel enabled check."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )

@swallow_all_exceptions(logger)
def main():
    """Initialize text-generation-inference server and client."""
    extra_params = {}

    logger.info("Init Start.")
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--mlflow_model", type=str, dest="mlflow_model", required=True)
    parser.add_argument("--parameters", type=str, dest="parameters", required=False, default="{}")
    parser.add_argument("--task", type=str, dest=ArgumentLiterals.TASK, required=True, choices=TEXT_TOKEN_TASKS)
    parser.add_argument("--data", type=str, dest=ArgumentLiterals.DATA, required=True)
    parser.add_argument("--label-column-name", type=lambda x: x.split(","),
                        dest=ArgumentLiterals.LABEL_COLUMN_NAME, required=False, default=None)
    parser.add_argument("--input-column-names",
                        type=lambda x: [i.strip() for i in x.split(",") if i and not i.isspace()],
                        dest=ArgumentLiterals.INPUT_COLUMN_NAMES, required=False, default=None)
    parser.add_argument("--batch-size", type=int, dest=ArgumentLiterals.BATCH_SIZE, required=False, default=None)
    parser.add_argument("--predictions", type=str, dest=ArgumentLiterals.PREDICTIONS, required=True)
    parser.add_argument("--ground-truth", type=str, dest=ArgumentLiterals.GROUND_TRUTHS, required=True)
    parser.add_argument("--performance-metadata", type=str, dest=ArgumentLiterals.PERFORMANCE_METADATA,
                        required=False, default=None)
    parser.add_argument("--prediction-probabilities", type=str, dest=ArgumentLiterals.PREDICTION_PROBABILITIES,
                    required=False, default=None)

    args, unknown_args = parser.parse_known_args()
    logger.info(f"Distributed Type: {distributed_state.distributed_type}")
    try:
        tensor_parallel, num_replicas = get_smart_defaults(args.mlflow_model)
    except Exception as e:
        exception = get_azureml_exception(ModelLoadingException, ModelPredictionInternalError, e,
                                              wrap_azureml_ex=False, error=repr(e))
        log_traceback(exception, logger)
        raise exception
    logger.info(f"Setting Num Replicas to: {num_replicas} and Tensor Parallel to {tensor_parallel}")
    os.environ["NUM_REPLICAS"] = str(num_replicas)
    os.environ["TENSOR_PARALLEL"] = str(tensor_parallel)

    data_path = args.data

    logger.info(f"Torch Current Device Count:{torch.cuda.device_count()}")
    logger.info(f"Got Params: {args.parameters}")
    extra_params.update(json.loads(args.parameters))
    
    logger.info(f"Got Model Path: {args.mlflow_model}")
    task_type = args.task
    input_column_names, label_column_name, extra_y_test_cols = validate_and_get_columns(vars(args))

    try:
        _init_cuda_visible_devices()

        abs_mlmodel_path = os.path.join(
            args.mlflow_model, ModelPath.MLMODEL_PATH
        )
        mlmodel = {}
        if abs_mlmodel_path and os.path.exists(abs_mlmodel_path):
            with open(abs_mlmodel_path) as f:
                mlmodel = yaml.safe_load(f)
        if os.path.exists(os.path.join(args.mlflow_model, ModelPath.DEFAULT_MLFLOW_MODEL_PATH)):
            model_path = os.path.join(
                args.mlflow_model,
                ModelPath.DEFAULT_MLFLOW_MODEL_PATH,
            )
            config_path = os.path.join(model_path, "config.json")
            tokenizer_path = os.path.join(
                args.mlflow_model, ModelPath.DEFAULT_TOKENIZER_PATH
            )
        else:
            model_path = os.path.join(args.mlflow_model, ModelPath.DEPRECATED_MLFLOW_MODEL_PATH)
            config_path = os.path.join(
                args.mlflow_model, ModelPath.DEPRECATED_MLFLOW_CONFIG_PATH, "config.json"
            )
            if not os.path.exists(config_path):
                config_path = os.path.join(model_path, "config.json")
            tokenizer_path = os.path.join(
                args.mlflow_model, ModelPath.DEPRECATED_MLFLOW_TOKENIZER_PATH
            )
            if not os.path.exists(tokenizer_path):
                tokenizer_path = model_path
        inference_config = None
        if os.path.exists(os.path.join(args.mlflow_model, ModelPath.INFERENCE_CONFIG_PATH)):
            inference_config = os.path.join(args.mlflow_model, ModelPath.INFERENCE_CONFIG_PATH)
        engine_config, task_config, default_generator_configs, task_type, model_info = build_configs_from_model(
            mlmodel,
            model_path,
            config_path,
            tokenizer_path,
            inference_config
        )

        config = {
            "engine": engine_config,
            "task": task_config,
        }
        enable_character_counts, enable_token_counts = False, False
        if extra_params.get("token_count_per_sample", False):
            enable_token_counts = True
            extra_params.pop("token_count_per_sample")
        if extra_params.get("char_count_per_sample", False):
            enable_character_counts = True
            extra_params.pop("char_count_per_sample")
        tokenizer = None
        if (task_type in TEXT_TOKEN_TASKS and enable_token_counts) or (task_type == SupportedTask.CHAT_COMPLETION or task_type == TaskType.CONVERSATIONAL):
            tokenizer = load_tokenizer(engine_config["tokenizer"], engine_config["ml_model_info"].get("hf_tokenizer_class", "AutoTokenizer"))

        g_fmscorer = FMScore(config)
        g_fmscorer.init()
        if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
            for k, v in os.environ.items():
                logger.info(f"env: {k} = {v}")
            logger.info(
                f"updated default_generator_configs: "
                f"{default_generator_configs}"
            )

    except Exception as e:
        exception = get_azureml_exception(ModelLoadingException, BadModel, e, error=repr(e))
        log_traceback(exception, logger)
        raise exception


    try:
        data = load_data(task_type, data_path, label_column_name, input_column_names, extra_y_test_cols, args.batch_size)
    except Exception as e:
        exception = get_azureml_exception(DataLoaderException, BadInputData, e, error=repr(e))
        log_traceback(exception, logger)
        raise exception
    full_data = [(x, y) for x, y in data]
    logger.info(f"Dataset size: {len(full_data)}")
    predictor = Predictor(g_fmscorer, task_type, extra_params, num_replicas, label_column_name, tokenizer, extra_y_test_cols)
    collated_res = [{} for i in range(distributed_state.num_processes)]
    with distributed_state.split_between_processes(full_data) as proc_data:
        y_pred_proc, y_test_proc, y_perf_proc, y_pred_proba = predictor.predict(proc_data)
        proc_res = {"predictions": y_pred_proc, "ground_truth": y_test_proc, "perf": y_perf_proc, "pred_probas": y_pred_proba}
        dist.all_gather_object(object_list=collated_res, obj=proc_res)
    logger.info("Waiting for all processes.....")
    distributed_state.wait_for_everyone()
    logger.info(f"Collated Results Lengths: {[len(i) for i in collated_res]}")
    y_pred_df, y_test_df, y_perf_df, y_pred_proba_df = _gather_predictions(collated_res)

    if task_type != SupportedTask.CHAT_COMPLETION and task_type != TaskType.CONVERSATIONAL:
        y_pred_df.columns = ["predictions"]
    ground_truth_columns = [label_column_name]
    if extra_y_test_cols is not None:
        ground_truth_columns += extra_y_test_cols
    y_test_df.columns = ground_truth_columns[:]

    if distributed_state.is_main_process:
        y_pred_df.to_json(args.predictions, orient="records", lines=True)
        y_test_df.to_json(args.ground_truths, orient="records", lines=True)
        y_perf_df.to_json(args.performance_metadata, orient="records", lines=True)
        y_pred_proba_df.to_json(args.prediction_probabilities, orient="records", lines=True)
    return


if __name__ == "__main__":
    main()