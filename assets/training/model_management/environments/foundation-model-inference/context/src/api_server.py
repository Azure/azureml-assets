# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Run script to infer."""
# flake8: noqa

from contextlib import asynccontextmanager
import dataclasses
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import yaml
import shutil
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, AnalyzeImageOptions, ImageData
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import HeadersPolicy
from azureml.ai.monitoring import Collector
from constants import EngineName, ServerSetupParams, SupportedTask, TaskType
from model_config_factory import ModelConfigFactory
from fm_score import FMScore
from logging_config import configure_logger
from managed_inference import MIRPayload
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from utils import box_logger, map_env_vars_to_vllm_server_args
from configs import EngineConfig, TaskConfig
from transformers import AutoTokenizer
from engine import InferenceResult
import api_server_setup

# openai api imports
import argparse
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
import uvicorn
from importlib import resources as impresources
from api_server_setup.protocol import ChatCompletionResponseChoice, ChatCompletionResponse, CompletionResponse, CompletionResponseChoice
from vllm.entrypoints.openai.protocol import ChatMessage, UsageInfo
from torch.multiprocessing import set_start_method
from vllm.utils import random_uuid
from typing import AsyncGenerator, Generator, Optional, Dict, Union, List

logger = configure_logger(__name__)

# AACS
g_aacs_threshold = int(os.environ.get("CONTENT_SAFETY_THRESHOLD", 2))
g_aacs_client = None

# default values
TIMEOUT_KEEP_ALIVE = 5  # seconds.
DEVICE_COUNT = torch.cuda.device_count()
MLMODEL_PATH = "mlflow_model_folder/MLmodel"
DEPRECATED_MLFLOW_MODEL_PATH = "mlflow_model_folder/data/model"
DEFAULT_MLFLOW_MODEL_PATH = "mlflow_model_folder/model"
DEFAULT_TOKENIZER_PATH = "mlflow_model_folder/components/tokenizer"
task_type = SupportedTask.TEXT_GENERATION
g_fmscorer: FMScore = None
g_model_signature = None
g_served_model = None
g_engine_config = None

# For apis
app = FastAPI(openapi_url="/swagger.json", docs_url="/swagger.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# # metrics tracking
g_collector = Collector(
    name="inference_metrics",
    on_error=lambda e: logger.info("ex:{}".format(e)),
)

# For swagger
def custom_openapi():
    """Populate openapi values for swagger."""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AzureML Foundation Model Inference Server",
        version="1.0.0",
        summary="A server for inferencing AzureML Foundation Models.",
        routes=app.routes,
    )

    swagger_file = (impresources.files(api_server_setup) / 'openapi.json')
    with open(swagger_file) as f:
        d = json.load(f) 
    openapi_schema["paths"] = d
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi


# region AACS
class CsChunkingUtils:
    """Cs chunking utils."""

    def __init__(self, chunking_n=1000, delimiter="."):
        """Init function."""
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        """Chunk strings in a given length."""
        return (string[0 + i : length + i] for i in range(0, len(string), length))

    def split_by(self, input):
        """Split the input."""
        max_n = self.chunking_n
        split = [e + self.delimiter for e in input.split(self.delimiter) if e]
        ret = []
        buffer = ""

        for i in split:
            # if a single element > max_n, chunk by max_n
            if len(i) > max_n:
                ret.append(buffer)
                ret.extend(list(self.chunkstring(i, max_n)))
                buffer = ""
                continue
            if len(buffer) + len(i) <= max_n:
                buffer = buffer + i
            else:
                ret.append(buffer)
                buffer = i

        if len(buffer) > 0:
            ret.append(buffer)
        return ret


def analyze_response(response: Union[Response, Dict]):
    """Analyze response."""
    severity = 0

    if response.hate_result is not None:
        logger.info("Hate severity: {}".format(response.hate_result.severity))
        severity = max(severity, response.hate_result.severity)
    if response.self_harm_result is not None:
        logger.info("SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        logger.info("Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        logger.info("Violence severity: {}".format(response.violence_result.severity))
        severity = max(severity, response.violence_result.severity)

    return severity


def analyze_text(text: str):
    """Analyze text."""
    global g_aacs_client
    # Chunk text
    logger.info("Analyzing ...")
    if (not text) or (not text.strip()):
        return 0
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    result = [analyze_response(g_aacs_client.analyze_text(AnalyzeTextOptions(text=i))) for i in split_text]
    severity = max(result)
    logger.info(f"Analyzed, severity {severity}")

    return severity


def analyze_image(image_in_byte64: str) -> int:
    """Analyze image severity using azure content safety service.

    :param image_in_byte64: image in base64 format
    :type image_in_byte64: str
    :return: maximum severity of all categories
    :rtype: int
    """
    print("Analyzing image...")
    if image_in_byte64 is None:
        return 0
    request = AnalyzeImageOptions(image=ImageData(content=image_in_byte64))
    safety_response = g_aacs_client.analyze_image(request)
    severity = analyze_response(safety_response)
    print(f"Image Analyzed, severity {severity}")
    return severity


def _check_data_type_from_model_signature(key: str) -> str:
    """Check key data type from model signature.

    :param key: key of data (to analyze by AACS) in model input or output
    :type key: str
    :return: data type of key from model signature else return "str"
    :rtype: str
    """
    if g_model_signature is None or key is None:
        return "str"
    input_schema = g_model_signature["inputs"]
    output_schema = g_model_signature["outputs"]

    def _get_type(schema):
        if type(schema) == str:
            schema = json.loads(schema)

        for item in schema:
            if item["name"] == key:
                return item["type"]
        return None

    return _get_type(input_schema) or _get_type(output_schema) or "str"


def iterate(obj, current_key=None):
    """Iterate through obj and check content severity."""
    if isinstance(obj, dict):
        severity = 0
        for key, value in obj.items():
            obj[key], value_severity = iterate(value, current_key=key)
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        severity = 0
        for idx in range(len(obj)):
            obj[idx], value_severity = iterate(obj[idx], current_key=current_key)
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, pd.DataFrame):
        severity = 0
        columns = list(obj.columns)
        for i in range(obj.shape[0]):  # iterate over rows
            for j in range(obj.shape[1]):  # iterate over columns
                obj.at[i, j], value_severity = iterate(obj.at[i, j], current_key=columns[j])
                severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, str):
        if current_key and _check_data_type_from_model_signature(current_key) == "binary":
            severity = analyze_image(obj)
        else:
            severity = analyze_text(obj)
        if severity > g_aacs_threshold:
            return "", severity
        else:
            return obj, severity
    else:
        return obj, 0


def get_safe_response(result: Union[Dict, CompletionResponse, ChatCompletionResponse]):
    """Check if response is safe."""
    global g_aacs_client
    logger.info("Analyzing response...")
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")
    if not g_aacs_client:
        return jsonable_result

    result, severity = iterate(jsonable_result)
    logger.info(f"Response analyzed, severity {severity}")
    return result


def get_safe_input(input_data: Dict):
    """Check if input is safe."""
    global g_aacs_client
    if not g_aacs_client:
        return input_data, 0
    logger.info("Analyzing input...")
    result, severity = iterate(input_data)
    logger.info(f"Input analyzed, severity {severity}")
    return result, severity


def get_aacs_access_key():
    """Get aacs access key."""
    key = os.environ.get("CONTENT_SAFETY_KEY")

    if key:
        return key

    uai_client_id = os.environ.get("UAI_CLIENT_ID")
    if not uai_client_id:
        raise RuntimeError(
            "Cannot get AACS access key, both UAI_CLIENT_ID and " "CONTENT_SAFETY_KEY are not set, exiting...",
        )

    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group_name = os.environ.get("RESOURCE_GROUP_NAME")
    aacs_account_name = os.environ.get("CONTENT_SAFETY_ACCOUNT_NAME")
    from azure.identity import ManagedIdentityCredential
    from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

    credential = ManagedIdentityCredential(client_id=uai_client_id)
    cs_client = CognitiveServicesManagementClient(credential, subscription_id)
    key = cs_client.accounts.list_keys(
        resource_group_name=resource_group_name,
        account_name=aacs_account_name,
    ).key1

    return key


def aacs_setup():
    """Create an AACS endpoint for the server to check input and outputs."""
    global g_aacs_client
    AACS_error = None
    try:
        endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT", None)
        key = get_aacs_access_key()

        if not endpoint:
            raise Exception("CONTENT_SAFETY_ENDPOINT env not set for AACS.")
        if not key:
            raise Exception("CONTENT_SAFETY_KEY env not set for AACS.")

        # Create a Content Safety client
        headers_policy = HeadersPolicy()
        headers_policy.add_header("ms-azure-ai-sender", "fm-optimized-inference")
        g_aacs_client = ContentSafetyClient(
            endpoint,
            AzureKeyCredential(key),
            headers_policy=headers_policy,
        )
    except Exception as e:
        AACS_error = e

    return AACS_error

# endregion


def get_generator_params(params: dict):
    """Return accumulated generator params."""
    updated_params = {}
    # map 'max_gen_len' to 'max_new_tokens' if present
    if "max_gen_len" in params:
        logger.warning("max_gen_len is deprecated. Use max_new_tokens")
        params["max_new_tokens"] = params["max_gen_len"]
        del params["max_gen_len"]

    updated_params.update(params)
    return updated_params


def _init_cuda_visible_devices():
    import torch

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return

    if "NVIDIA_VISIBLE_DEVICES" in os.environ and os.environ["NVIDIA_VISIBLE_DEVICES"] != "all":
        # map the gpu ids to integers
        gpu_ids = os.environ["NVIDIA_VISIBLE_DEVICES"].split(",")
        gpu_ids = [str(i) for i in range(len(gpu_ids)) if gpu_ids[i] != "-1"]
    elif torch.cuda.is_available():
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    else:
        # if no GPU is available, don't set anything
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)


@app.get("/")
async def health():
    """Check if server is healthy. Used by the readiness probe to check server is healthy."""
    print("health")
    return "healthy"


@app.post("/score")
async def score(request: Request) -> Response:
    """Generate completion for the request.

    This endpoint uses the AzureML standard for inputs, different from the openai api endpoints.
    """
    data = await request.json()

    inference_results = None
    try:
        inference_results, result_dict = await send_request(data)
    except:
        if inference_results is None:
            return {}

    all_logged_results = f""" ### Inference Results ###\n \
Total Generation Time: {inference_results[0].inference_time_ms}\n \
Throughput (prompt/sec): {len(inference_results) / (inference_results[0].inference_time_ms / 1000):.2f}"""
    box_logger(all_logged_results)

    stats_dict = [vars(result) for result in inference_results]
    g_collector.collect(stats_dict)
    return get_safe_response(result_dict)


@app.post("/v1/completions")
@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request) -> Response:
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    data = await request.json()

    if task_type == SupportedTask.CHAT_COMPLETION:
        logger.info(f"Received chat completion request!")
        return await chat_inference(data)
    elif task_type == SupportedTask.TEXT_GENERATION:
        logger.info(f"Received text generation request!")
        return await text_gen_inference(data)
    else:
        raise Exception(
            json.dumps({"error": f"task_type {task_type} not available with completions API."})
            )


async def chat_inference(data: Dict) -> ChatCompletionResponse:
    """Format data to send to a chat model."""
    global g_served_model

    new_data = {}
    parameters = {}
    for d in data:
        new_d = {}
        if d != 'model' and d != 'messages':
            new_d[d] = data[d]

            parameters.update({d: data[d]})
    new_data.update({'task_type': task_type})
    if 'prompt' in data or 'messages' not in data:
        raise Exception(
            json.dumps(
                {
                    "error": (
                        "Expected input format: \n"
                        ' "messages": [{ \n'
                            ' "role": "system", \n'
                            ' "content": "You are a helpful assistant."}, \n'
                            '{\n'
                               ' "role": "user", \n'
                               ' "content": "Hello!" \n'
                            ' }]\n'

                    )
                }
            )
        )
    else:
        raw_prompt = data["messages"]
        messages = {'input_data': {'input_string': raw_prompt, 'parameters': parameters}}
    new_data.update(messages)
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    tokenizer = AutoTokenizer.from_pretrained(g_engine_config["model_id"])
    num_prompt_tokens = len(tokenizer.apply_chat_template(raw_prompt, tokenize=True))

    inference_results = None
    try:
        inference_results, result_dict = await send_request(new_data)
    except:
        if inference_results is None:
            choice_data = ChatCompletionResponseChoice(
                finish_reason="content_filter"
            )
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                total_tokens=num_prompt_tokens
            )
            response = ChatCompletionResponse(
                id=request_id,
                created=created_time,
                model=g_served_model,
                choices=[choice_data],
                usage=usage,
            )
            return get_safe_response(response)

    choices = []
    for result in inference_results:
        choice_data = ChatCompletionResponseChoice(
            index=result.prompt_num,
            message=ChatMessage(role="assistant", content=result.response)
        )
        choices.append(choice_data)  

    num_generated_tokens = sum(
        len(output.generated_tokens) for output in inference_results)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=g_served_model,
        choices=choices,
        usage=usage,
    )
    return get_safe_response(response)

async def text_gen_inference(data: Dict) -> CompletionResponse:
    """Format data to send to a text generation model."""
    global g_served_model

    new_data = {}
    parameters = {}
    for d in data:
        new_d = {}
        if d != 'model' and d != 'prompt':
            new_d[d] = data[d]

            parameters.update({d: data[d]})
    new_data.update({'task_type': task_type})
    if 'messages' in data or 'prompt' not in data:
        raise Exception(
            json.dumps(
                {
                    "error": (
                        "Expected input format as string or list of strings: \n"
                        ' "prompt": "My favorite color is"\n'
                        ' or "prompt": "[My favorite color is, "The meaning of life is"]"\n'

                    )
                }
            )
        )
    else:
        raw_prompt = data["prompt"]
        if type(raw_prompt) is str:
            raw_prompt = [raw_prompt]
        messages = {'input_data': raw_prompt, 'params': parameters}
    new_data.update(messages)
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    num_prompt_tokens = 0
    tokenizer = AutoTokenizer.from_pretrained(g_engine_config["model_id"])  
    for prompt in raw_prompt:
        num_prompt_tokens += len(tokenizer.encode(prompt))

    inference_results = None
    try:
        inference_results, result_dict = await send_request(new_data)
    except:
        if inference_results is None:
            choice_data = CompletionResponseChoice(
                finish_reason="content_filter"
            )           
            usage = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=0,
                total_tokens=num_prompt_tokens
            )
            response = CompletionResponse(
                id=request_id,
                created=created_time,
                model=g_served_model,
                choices=[choice_data],
                usage=usage,
            )
            return get_safe_response(response)

    choices = []
    for result in inference_results:
        choice_data = CompletionResponseChoice(
            index=result.prompt_num,
            text=result.response,
        )
        choices.append(choice_data)

    num_generated_tokens = sum(
        len(output.generated_tokens) for output in inference_results)
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=g_served_model,
        choices=choices,
        usage=usage,
    )
    return get_safe_response(response)


async def send_request(data: Dict) -> (List[InferenceResult], Dict):
    """Run for inference data provided."""
    global g_fmscorer
    global task_type
    global g_engine_config

    try:
        data, severity = get_safe_input(data)
        if severity > g_aacs_threshold:
            logger.warning(
                f"Input severity ({severity}) greater than aacs threshold " f"({g_aacs_threshold}).",
            )
            return {}

        data.update({"task_type": task_type})

        payload = MIRPayload.from_dict(data)
        payload.update_params(get_generator_params(payload.params))
        logger.info(
            f"Processing new request with parameters: {payload.params}",
        )

        results = {}
        inference_results = None
        if task_type == SupportedTask.CHAT_COMPLETION:
            payload.convert_query_to_list()
        if g_engine_config["engine_name"] == EngineName.MII or g_engine_config["engine_name"] == EngineName.MII_V1:
            try:
                inference_results = await g_fmscorer.run_async(payload)
            except Exception as e:
                raise Exception(
                    json.dumps({"error": "Error in processing request", "exception": str(e)}))
        else:
            inference_results = g_fmscorer.run(payload)

        if task_type == SupportedTask.CHAT_COMPLETION:
            outputs = {str(i): res.response for i, res in enumerate(inference_results)}
            results = {
                "output": f"{outputs['0']}",
            }  # outputs will only have one key for chat-completion
        elif task_type == SupportedTask.TEXT_TO_IMAGE:
            results = [dataclasses.asdict(res.response) for res in inference_results]
        else:
            assert task_type == SupportedTask.TEXT_GENERATION and isinstance(
                payload.query,
                list,
            ), "query should be a list for text-generation"
            # With models converted to transformers flavor from hftransformersv2 flavor, 
            # the output is a list of strings
            if payload.is_preview_format:
                # TODO: Remove this once all models are updated to new format
                outputs = {str(i): res.response for i, res in enumerate(inference_results)}
                results = pd.DataFrame([outputs])
            else:
                results = [res.response for res in inference_results]


        return inference_results, results

    except Exception as e:
        logger.exception(e)
        raise Exception(
            json.dumps({"error": "Error in processing request", "exception": str(e)})
        )


@asynccontextmanager  
async def lifespan(app: FastAPI) -> Generator:  
    """Initialize and shutdown events for each worker that is spawned in the main function."""
    global g_engine_config  
    global g_served_model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), DEFAULT_MLFLOW_MODEL_PATH
        )

    g_served_model = model_path

    # setup aacs
    AACS_error = aacs_setup()

    init_error = init_server(model_path, AACS_error)
    if init_error:
        logger.exception(init_error)
        raise init_error

    if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
        box_logger(f"PID[{str(os.getpid())}] Inference server successfully started with model {g_served_model}")

    yield  # Add any necessary shutdown events after yield statement

    if g_engine_config["engine_name"] == EngineName.MII or g_engine_config["engine_name"] == EngineName.MII_V1:
            try:
                await g_fmscorer.shutdown_async()
            except Exception as e:
                raise Exception(
                    json.dumps({"error": "Error in processing request", "exception": str(e)}))


# Assign the lifespan event handler to the FastAPI instance  
app.router.lifespan_context = lifespan 


def init_server(model_path: str, AACS_error: Union[None, Exception]):
    """Initialize text-generation-inference server and client."""
    global task_type
    global g_fmscorer
    global g_engine_config
    global g_model_signature
    try:

        tokenizer_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR", ""),
            DEFAULT_TOKENIZER_PATH,
        )

        # Maintain Backwards Compatibility with old file structure
        if not os.path.exists(model_path):
            model_path = os.path.join(
                os.getenv("AZUREML_MODEL_DIR", ""),
                DEPRECATED_MLFLOW_MODEL_PATH,
            )
            tokenizer_path = model_path

        config_path = os.path.join(model_path, "config.json")

        default_engine = EngineName.VLLM
        tensor_parallel = os.getenv("TENSOR_PARALLEL", None)
        if tensor_parallel:
            tensor_parallel = int(tensor_parallel)
        g_engine_config = {
            "engine_name": os.getenv("ENGINE_NAME", default_engine),
            "model_id": model_path,
            "tokenizer": tokenizer_path,
            "tensor_parallel": tensor_parallel,
        }

        _init_cuda_visible_devices()

        abs_mlmodel_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR", ""),
            MLMODEL_PATH,
        )
        mlmodel = {}
        if abs_mlmodel_path and os.path.exists(abs_mlmodel_path):
            with open(abs_mlmodel_path) as f:
                mlmodel = yaml.safe_load(f)

        default_generator_configs = ""
        task_config = None
        if mlmodel:
            flavors = mlmodel.get("flavors", {})
            g_model_signature = mlmodel.get("signature", None)
            if "transformers" in flavors:
                # New MLModel format with transformers flavor
                task_type = flavors["transformers"]["task"]

                model_generator_configs = flavors["transformers"].get(
                    "generator_config",
                    {},
                )
                if task_type not in (
                    SupportedTask.TEXT_GENERATION,
                    SupportedTask.CHAT_COMPLETION,
                ):
                    raise Exception(f"Unsupported task_type {task_type}")

                # update default gen configs with model configs
                default_generator_configs = get_generator_params(
                    model_generator_configs,
                )
            elif "hftransformersv2" in flavors:
                # Old MLModel format with hftransformersv2 flavor
                # TODO: Remove this once all models are updated to new format
                task_type = flavors["hftransformersv2"]["task_type"]
                model_generator_configs = flavors["hftransformersv2"].get(
                    "generator_config",
                    {},
                )

                if task_type not in (
                    SupportedTask.TEXT_GENERATION,
                    SupportedTask.CHAT_COMPLETION,
                ):
                    raise Exception(f"Unsupported task_type {task_type}")

                # update default gen configs with model configs
                default_generator_configs = get_generator_params(
                    model_generator_configs,
                )
            elif "python_function" in flavors:

                task_type = mlmodel["metadata"]["base_model_task"]
                if task_type not in (TaskType.TEXT_TO_IMAGE):
                    raise Exception(f"Unsupported task_type {task_type}")

                model_type = mlmodel["metadata"].get("model_type", "")

                model_config_builder = ModelConfigFactory.get_config_builder(task=task_type, model_type=model_type)
                g_engine_config.update(
                                        {
                                            "engine_name": model_config_builder.engine,
                                            "mii_config": model_config_builder.get_optimization_config(),
                                            "custom_model_config_builder": model_config_builder,
                                            "model_id": os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_config_builder.MLFLOW_MODEL_PATH),
                                            "tokenizer": os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_config_builder.MLFLOW_MODEL_PATH, "tokenizer"),
                                            "tensor_parallel": model_config_builder.tensor_parallel
                                        }
                                    )
                task_config = model_config_builder.get_task_config()

        if g_engine_config["engine_name"] in [EngineName.MII, EngineName.MII_V1] and "mii_config" not in g_engine_config:
            mii_engine_config = {
                "deployment_name": os.getenv("DEPLOYMENT_NAME", "llama-deployment"),
                "mii_configs": {},
            }

            g_engine_config["mii_config"] = mii_engine_config

        if g_engine_config["engine_name"] == EngineName.VLLM:
            model_config = {}
            vllm_config = map_env_vars_to_vllm_server_args()
            if config_path and os.path.exists(config_path):
                with open(config_path) as config_content:
                    model_config = json.load(config_content)
            g_engine_config["vllm_config"] = vllm_config
            g_engine_config["model_config"] = model_config

        task_config = {
            "task_type": TaskType.CONVERSATIONAL
            if task_type == SupportedTask.CHAT_COMPLETION
            else TaskType.TEXT_GENERATION,
        } if task_config is None else task_config

        config = {
            "engine": g_engine_config,
            "task": task_config,
        }

        g_fmscorer = FMScore(config)
        g_fmscorer.init()
        if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
            for k, v in os.environ.items():
                logger.info(f"env: {k} = {v}")
            logger.info(
                f"updated default_generator_configs: " f"{default_generator_configs}",
            )
            if AACS_error:
                logger.warning(f"AACS was not configured. Content moderation bypassed in setup. Error {AACS_error}")
        return None
    except Exception as e:
        return Exception(f"Error in creating client or server: {e}")


if __name__ == "__main__":
    """Initialize text-generation-inference server and client."""

    # Start fast api server
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)

    args = parser.parse_args()

    set_start_method('spawn')

    workers = int(os.getenv("WORKER_COUNT", ServerSetupParams.DEFAULT_WORKER_COUNT))
    log_level = str(os.getenv("AZUREML_LOG_LEVEL", "warning"))
    log_level = log_level.lower()

    print("starting server with host", args.host)
    print("starting server with port", args.port)

    uvicorn.run("__main__:app",
                host=args.host,
                port=args.port,
                log_level=log_level,
                workers=workers,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
