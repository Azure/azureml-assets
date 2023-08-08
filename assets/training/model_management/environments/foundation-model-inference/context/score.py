# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run script to infer."""

import asyncio
import json
import os
import psutil
import requests
import time
import yaml
import logging
import pandas as pd
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from typing import List, Dict, Any, Tuple, Union
from text_generation import Client

from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from aiolimiter import AsyncLimiter
from azure.core.pipeline.policies import HeadersPolicy

# Configure logger
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
format_str = ("%(asctime)s [%(module)s] %(funcName)s "
              "%(lineno)s: %(levelname)-8s [%(process)d] %(message)s")
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

PORT = 80
LOCAL_HOST_URI = f"http://0.0.0.0:{PORT}"

TEXT_GEN_LAUNCHER_PROCESS_NAME = "text-generation-launcher"

# model init env vars
MODEL_ID = "MODEL_ID"
SHARDED = "SHARDED"
NUM_SHARD = "NUM_SHARD"
QUANTIZE = "QUANTIZE"
DTYPE = "DTYPE"
TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"
MAX_CONCURRENT_REQUESTS = "MAX_CONCURRENT_REQUESTS"
MAX_BEST_OF = "MAX_BEST_OF"
MAX_STOP_SEQUENCES = "MAX_STOP_SEQUENCES"
MAX_INPUT_LENGTH = "MAX_INPUT_LENGTH"
MAX_TOTAL_TOKENS = "MAX_TOTAL_TOKENS"

# default values
DEFAULT_MAX_INPUT_LENGTH = 2048
DEFAULT_MAX_TOTAL_TOKENS = 4096

# client init env vars
CLIENT_TIMEOUT = "TIMEOUT"
MAX_REQUEST_TIMEOUT = 90  # 90s

# AACS
aacs_threshold = int(os.environ.get("CONTENT_SAFETY_THRESHOLD", 2))
aacs_client = None


class SupportedTask:
    """Supported tasks by text-generation-inference."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"


SUPPORTED_INFERENCE_PARAMS = {
    # Activate logits sampling
    "do_sample": {"type": bool, "default": True},
    # Maximum number of generated tokens
    "max_new_tokens": {"type": int, "default": 256},
    # Generate best_of sequences & return the one with highest token logprobs
    "best_of": {"type": int, "optional": True},
    # 1.0 means no penalty. See
    # [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.
    "repetition_penalty": {"type": int, "optional": True},
    # Whether to prepend the prompt to the generated text
    "return_full_text": {"type": bool, "default": True},
    # The value used to module the logits distribution.
    "seed": {"type": int, "optional": True},
    # Stop generating tokens if a member of `stop_sequences` is generated
    "stop_sequences": {"type": list, "default": []},
    # Random sampling seed
    "temperature": {"type": float, "optional": True},
    # The number of highest probability vocabulary tokens to keep for
    # top-k-filtering.
    "top_k": {"type": int, "optional": True},
    # If set to < 1, only the smallest set of most probable tokens with
    # probabilities that add up to
    # `top_p` or higher are kept for generation.
    "top_p": {"type": float, "optional": True},
    # Truncate inputs tokens to the given size
    "truncate": {"type": int, "optional": True},
    # Typical Decoding mass. See:
    # [Typical Decoding for Natural Language Generation]\
    # (https://arxiv.org/abs/2202.00666) for more information
    "typical_p": {"type": float, "optional": True},
    # Watermarking with [A Watermark for Large Language Models]\
    # (https://arxiv.org/abs/2301.10226)
    "watermark": {"type": bool, "default": False},
    # Get decoder input token logprobs and ids
    "decoder_input_details": {"type": bool, "default": False},
}

# default values
MLMODEL_PATH = "mlflow_model_folder/MLmodel"
DEFAULT_MODEL_ID_PATH = "mlflow_model_folder/data/model"
client = None
task_type = SupportedTask.TEXT_GENERATION
default_generator_configs = {
    k: v["default"] for k, v in SUPPORTED_INFERENCE_PARAMS.items() if
    "default" in v
}


def is_server_healthy():
    """Periodically checks if server is up and running."""
    # use psutil to go through active process
    WAIT_TIME = 20
    RETRY_COUNT = 5
    count = 0
    while count < RETRY_COUNT and TEXT_GEN_LAUNCHER_PROCESS_NAME not in [
        p.name() for p in psutil.process_iter()
    ]:
        logger.warning(
            f"Process {TEXT_GEN_LAUNCHER_PROCESS_NAME} is not running. "
            f"Sleeping for {WAIT_TIME}s and retrying"
        )
        time.sleep(WAIT_TIME)
        count += 1
    if count >= RETRY_COUNT:
        total_dur = RETRY_COUNT * WAIT_TIME
        raise Exception(
            f"Sever process not running after waiting for {total_dur}. "
            f"Terminating"
        )

    logger.info(
        f"Server process {TEXT_GEN_LAUNCHER_PROCESS_NAME} running. "
        f"Hitting endpoint with 5s delay"
    )
    time.sleep(5)

    payload_dict = {"inputs": "Meaning of life is",
                    "parameters": {"max_new_tokens": 2}}

    json_str = json.dumps(payload_dict)

    try:
        response = requests.post(
            url=LOCAL_HOST_URI,
            data=json_str,
            headers={"Content-Type": "application/json"},
        )
        logger.info(f"response status code: {response.status_code}")
        if response.status_code == 200 or response.status_code == 201:
            return True
    except Exception as e:
        logger.warning(f"Test request failed. Error {e}")
    return False


# ACS START


class AsyncRateLimitedOpsUtils:
    """
    Util function for async rate limiter.

    1000 requests / 10 seconds. Limiting to 800 request per 10 secods
    limiting to 1000 concurrent requests
    """

    def __init__(
            self,
            ops_count=800,
            ops_seconds=10,
            concurrent_ops=1000,
            thread_max_workers=1000,
    ):
        """Init function."""
        self.limiter = AsyncLimiter(ops_count, ops_seconds)
        self.semaphore = asyncio.Semaphore(value=concurrent_ops)
        # need thread pool executor for sync function
        self.executor = ThreadPoolExecutor(max_workers=thread_max_workers)

    def get_limiter(self):
        """Return limiter."""
        return self.limiter

    def get_semaphore(self):
        """Rreturn semaphore."""
        return self.semaphore

    def get_executor(self):
        """Return executor."""
        return self.executor


async_rate_limiter = AsyncRateLimitedOpsUtils()


class CsChunkingUtils:
    """Cs chunking utils."""

    def __init__(self, chunking_n=1000, delimiter="."):
        """Init function."""
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        """Chunk strings in a given length."""
        return (string[0 + i: length + i] for i in
                range(0, len(string), length))

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


async def async_analyze_text_task(client, request):
    """Analyze text task."""
    loop = asyncio.get_event_loop()
    executor = async_rate_limiter.get_executor()
    sem = async_rate_limiter.get_semaphore()
    await sem.acquire()
    async with async_rate_limiter.get_limiter():
        response = await loop.run_in_executor(executor, client.analyze_text,
                                              request)
        sem.release()
        severity = analyze_response(response)
        return severity


def analyze_response(response):
    """Analyze response."""
    severity = 0

    if response.hate_result is not None:
        logger.info("Hate severity: {}".format(response.hate_result.severity))
        severity = max(severity, response.hate_result.severity)
    if response.self_harm_result is not None:
        logger.info(
            "SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        logger.info(
            "Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        logger.info(
            "Violence severity: {}".format(response.violence_result.severity))
        severity = max(severity, response.violence_result.severity)

    return severity


def analyze_text_async(text):
    """Analyze text async."""
    global aacs_client
    # Chunk text
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    tasks = []
    for i in split_text:
        request = AnalyzeTextOptions(text=i)
        tasks.append(async_analyze_text_task(aacs_client, request))

    done, pending = asyncio.get_event_loop().run_until_complete(
        asyncio.wait(tasks, timeout=60)
    )

    if len(pending) > 0:
        # not all task finished, assume failed
        return 6

    return max([d.result() for d in done])


def analyze_text(text):
    """Analyze text."""
    global aacs_client
    # Chunk text
    logger.info("Analyzing ...")
    if (not text) or (not text.strip()):
        return 0
    chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
    split_text = chunking_utils.split_by(text)

    result = [
        analyze_response(aacs_client.analyze_text(AnalyzeTextOptions(text=i)))
        for i in split_text
    ]
    severity = max(result)
    logger.info(f"Analyzed, severity {severity}")

    return severity


def iterate(obj):
    """Iterate through obj and check content severity."""
    if isinstance(obj, dict):
        severity = 0
        for key, value in obj.items():
            obj[key], value_severity = iterate(value)
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        severity = 0
        for idx in range(len(obj)):
            obj[idx], value_severity = iterate(obj[idx])
            severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, pd.DataFrame):
        severity = 0
        for i in range(obj.shape[0]):  # iterate over rows
            for j in range(obj.shape[1]):  # iterate over columns
                obj.at[i, j], value_severity = iterate(obj.at[i, j])
                severity = max(severity, value_severity)
        return obj, severity
    elif isinstance(obj, str):
        severity = analyze_text(obj)
        if severity > aacs_threshold:
            return "", severity
        else:
            return obj, severity
    else:
        return obj, 0


def get_safe_response(result):
    """Check if response is safe."""
    global aacs_client
    logger.info("Analyzing response...")
    jsonable_result = _get_jsonable_obj(result, pandas_orient="records")
    if not aacs_client:
        return jsonable_result

    result, severity = iterate(jsonable_result)
    logger.info(f"Response analyzed, severity {severity}")
    return result


def get_safe_input(input_data):
    """Check if input is safe."""
    global aacs_client
    if not aacs_client:
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
            "Cannot get AACS access key, both UAI_CLIENT_ID and "
            "CONTENT_SAFETY_KEY are not set, exiting..."
        )

    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    resource_group_name = os.environ.get("RESOURCE_GROUP_NAME")
    aacs_account_name = os.environ.get("CONTENT_SAFETY_ACCOUNT_NAME")
    from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
    from azure.identity import ManagedIdentityCredential

    credential = ManagedIdentityCredential(client_id=uai_client_id)
    cs_client = CognitiveServicesManagementClient(credential, subscription_id)
    key = cs_client.accounts.list_keys(
        resource_group_name=resource_group_name, account_name=aacs_account_name
    ).key1

    return key


# ACS END


def init():
    """Initialize text-generation-inference server and client."""
    global client
    global task_type
    global default_generator_configs
    global aacs_client

    try:
        logger.info("Setting up AACS")
        endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT", None)
        key = get_aacs_access_key()

        if not endpoint:
            raise Exception("CONTENT_SAFETY_ENDPOINT env not set for AACS.")
        if not key:
            raise Exception("CONTENT_SAFETY_KEY env not set for AACS.")

        # Create a Content Safety client
        headers_policy = HeadersPolicy()
        headers_policy.add_header("ms-azure-ai-sender", "llama")
        aacs_client = ContentSafetyClient(
            endpoint, AzureKeyCredential(key), headers_policy=headers_policy
        )
    except Exception as e:
        logger.error(
            f"AACS not configured. Bypassing content moderation. Error {e}")

    try:
        model_id = os.environ.get(MODEL_ID, DEFAULT_MODEL_ID_PATH)
        client_timeout = os.environ.get(CLIENT_TIMEOUT, MAX_REQUEST_TIMEOUT)

        for k, v in os.environ.items():
            logger.info(f"env: {k} = {v}")

        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", ""), model_id)
        abs_mlmodel_path = os.path.join(
            os.getenv("AZUREML_MODEL_DIR", ""), MLMODEL_PATH
        )
        mlmodel = {}
        if abs_mlmodel_path and os.path.exists(abs_mlmodel_path):
            with open(abs_mlmodel_path) as f:
                mlmodel = yaml.safe_load(f)

        if mlmodel:
            flavors = mlmodel.get("flavors", {})
            if "hftransformersv2" in flavors:
                task_type = flavors["hftransformersv2"]["task_type"]
                model_generator_configs = flavors["hftransformersv2"].get(
                    "generator_config", {}
                )
                logger.info(
                    f"model_generator_configs: {model_generator_configs}")
                if task_type not in (
                        SupportedTask.TEXT_GENERATION,
                        SupportedTask.CHAT_COMPLETION,
                ):
                    raise Exception(f"Unsupported task_type {task_type}")

                if task_type == SupportedTask.CHAT_COMPLETION:
                    default_generator_configs["return_full_text"] = False
                # update default gen configs with model configs
                default_generator_configs = get_generator_params(
                    model_generator_configs
                )
                logger.info(
                    f"updated default_generator_configs: "
                    f"{default_generator_configs}"
                )

        logger.info(
            f"Loading model from path {model_path} for task_type: {task_type}")
        logger.info(f"List model_path = {os.listdir(model_path)}")

        if MAX_INPUT_LENGTH not in os.environ:
            os.environ[MAX_INPUT_LENGTH] = str(DEFAULT_MAX_INPUT_LENGTH)

        if MAX_TOTAL_TOKENS not in os.environ:
            os.environ[MAX_TOTAL_TOKENS] = str(DEFAULT_MAX_TOTAL_TOKENS)

        if SHARDED not in os.environ and NUM_SHARD not in os.environ:
            # If neither are set (e.g. UI deployments), set sharded and/or
            # num_shard based on whether flash attention can be used or not.
            # SKUs like V100 can't use sharding (num_shard = 1)
            # If A100 or newer SKUs, set sharded to true and use .
            try:
                import torch
                major, minor = torch.cuda.get_device_capability()
                is_sm75 = major == 7 and minor == 5  # turing
                is_sm8x = major == 8 and minor >= 0  # ampere
                is_sm90 = major == 9 and minor == 0  # hopper

                if not (is_sm75 or is_sm8x or is_sm90):
                    # can't use flash attention & hence sharding.
                    # set number of shards to 1
                    logger.info(
                        f"Setting {NUM_SHARD} to 1 since flash attention "
                        f"can't be used on this GPU."
                    )
                    os.environ[NUM_SHARD] = str(1)
                else:
                    # number of shards is default to the number of GPUs
                    logger.info(f"Setting {SHARDED} to true.")
                    os.environ[SHARDED] = "true"
            except Exception:
                # CUDA not available. Set number of shards to 1
                logger.info(f"GPU unavailable. Setting {NUM_SHARD} to 1")
                os.environ[NUM_SHARD] = str(1)

        logger.info("Starting server")
        cmd = f"text-generation-launcher --model-id {model_path} &"
        os.system(cmd)
        time.sleep(20)

        WAIT_TIME = 60
        while not is_server_healthy():
            logger.info(
                f"Server not up. Waiting for {WAIT_TIME}s, "
                f"before querying again."
            )
            time.sleep(WAIT_TIME)
        logger.info("Server Started")

        # run nvidia-smi
        logger.info("###### GPU INFO ######")
        logger.info(os.system("nvidia-smi"))
        logger.info("###### GPU INFO ######")

        client = Client(
            LOCAL_HOST_URI, timeout=client_timeout
        )  # use deployment settings
        logger.info(f"Created Client: {client}")
    except Exception as e:
        raise Exception(f"Error in creating client or server: {e}") from e


def get_processed_input_data_for_chat_completion(dialog: List[str]) -> str:
    r"""Process chat completion input request.

    Taken from:
    https://github.com/facebookresearch/llama/blob/main/llama/generation.py

    example input:
    [
        {
            "role": "user",
            "content": "What is the tallest building in the world?"
        },
        {
            "role": "assistant",
            "content": "As of 2021, the Burj Khalifa in Dubai"
        },
        {
            "role": "user",
            "content": "and in Africa?"
        },
    ]
    example output:
    "[INST]What is the tallest building in the world?[/INST]
    As of 2021, the Burj Khalifa in Dubai\n
    [INST]and in Africa?[/INST]"
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    # fmt: off
    DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest \
assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, \
dangerous, or illegal content. Please ensure that your responses are \
socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, \
explain why instead of answering something not correct. If you don't know \
the answer to a question, please don't share false information."""

    # fmt: on

    def process_dialog(messages) -> Tuple[str, List[Tuple[str, str]], str]:
        """Process messages to get system prompt, user-assistant messages and
        last user message."""
        system_prompt = DEFAULT_SYSTEM_PROMPT
        user_assistant_messages = []  # list of (user, assistant) messages
        user_message = None  # current user message being processed
        last_user_message = None  # user prompt for which response is needed

        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            if role == "system" and i == 0:
                system_prompt = content
            elif role == "user":
                if i == len(messages) - 1:
                    last_user_message = content
                else:
                    user_message = content
            elif role == "assistant" and user_message is not None:
                user_assistant_messages.append((user_message, content))
                user_message = None

        return system_prompt, user_assistant_messages, last_user_message

    # ref: https://huggingface.co/spaces/huggingface-projects/\
    # llama-2-7b-chat/blob/main/model.py
    def format_chat_conv(
            message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(
                f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)

    sys_prompt, user_assistant_msgs, message = process_dialog(dialog)
    chat_conv = format_chat_conv(message, user_assistant_msgs, sys_prompt)
    return chat_conv


def get_request_data(request_string) -> (
        Tuple)[Union[str, List[str]], Dict[str, Any]]:
    """Process and validate inference request.

    return type for chat-completion: str, dict
    return type for text-generation: list, dict
    """
    global task_type
    try:
        data = json.loads(request_string)
        logger.info(f"data: {data}")
        inputs = data.get("input_data", None)

        input_data = []  # type: Union[str, List[str]]
        params = {}  # type: Dict[str, Any]

        if not isinstance(inputs, dict):
            raise Exception("Invalid input data")

        input_data = inputs["input_string"]
        params = inputs.get("parameters", {})

        if not isinstance(input_data, list):
            raise Exception("query is not a list")

        if not isinstance(params, dict):
            raise Exception("parameters is not a dict")

        if task_type == SupportedTask.CHAT_COMPLETION:
            logger.info("chat-completion task. Processing input data")
            input_data = get_processed_input_data_for_chat_completion(
                input_data)

        return input_data, params
    except Exception as e:
        raise Exception(
            json.dumps({
                "error": (
                    "Expected input format: \n"
                    '{"input_data": {"input_string": "<query>", '
                    '"parameters": {"k1":"v1", "k2":"v2"}}}.\n '
                    "<query> should be in below format:\n "
                    'For text-generation: ["str1", "str2", ...]\n'
                    'For chat-completion: [{"role":"user", "content": "str1"},'
                    '{"role": "assistant", "content": "str2"} ....]'
                ),
                "exception": str(e),
            })
        )


def add_and_validate_gen_params(params: dict, new_params: dict):
    """Add and validate inference params."""
    if not new_params or not isinstance(new_params, dict):
        return params
    for k, v in new_params.items():
        if k not in SUPPORTED_INFERENCE_PARAMS:
            logger.warning(f"Ignoring unsupported inference param {k}.")
        elif not isinstance(v, SUPPORTED_INFERENCE_PARAMS[k]["type"]):
            logger.warning(
                f"Ignoring inference param {k} as value passed is of type "
                f"{type(v)} and not of type "
                f"{SUPPORTED_INFERENCE_PARAMS[k]['type']}"
            )
        else:
            params[k] = v
    return params


def get_generator_params(params: dict):
    """Return accumulated generator params."""
    global default_generator_configs

    updated_params = {}
    updated_params.update(default_generator_configs)
    updated_params = add_and_validate_gen_params(updated_params, params)
    return updated_params


def run(data):
    """Run for inference data provided."""
    global client
    global task_type

    try:
        data, severity = get_safe_input(data)
        if severity > aacs_threshold:
            logger.warning(
                f"Input severity ({severity}) greater than aacs threshold "
                f"({aacs_threshold})."
            )
            return {}

        if client is None:
            raise Exception("Client is not initialized")

        query, params = get_request_data(data)
        params = get_generator_params(params)
        logger.info(
            f"generating response for input_string: {query}, "
            f"parameters: {params}"
        )

        if task_type == SupportedTask.CHAT_COMPLETION:
            time_start = time.time()
            response_str = client.generate(query, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"time_taken: {time_taken}")
            result_dict = {"output": f"{response_str}"}
            return get_safe_response(result_dict)

        assert task_type == SupportedTask.TEXT_GENERATION and isinstance(
            query, list
        ), "query should be a list for text-generation"

        results = {}
        for i, q in enumerate(query):
            time_start = time.time()
            response_str = client.generate(q, **params).generated_text
            time_taken = time.time() - time_start
            logger.info(f"query {i} - time_taken: {time_taken}")
            results[str(i)] = [f"{response_str}"]

        resp = pd.DataFrame(results)
        return get_safe_response(resp)

    except Exception as e:
        logger.exception(e)
        return json.dumps(
            {"error": "Error in processing request", "exception": str(e)})