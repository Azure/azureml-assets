# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run script to infer."""

import asyncio
import os
import json
import time
import torch
import mii
import numpy as np
import pandas as pd
import logging
import yaml
from typing import List, Dict, Any, Tuple, Union

from concurrent.futures import ThreadPoolExecutor
from mii.config import LoadBalancerConfig, ReplicaConfig
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azure.ai.contentsafety import ContentSafetyClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.contentsafety.models import AnalyzeTextOptions
from aiolimiter import AsyncLimiter
from azure.core.pipeline.policies import HeadersPolicy

model = None
logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)
format_str = "%(asctime)s [%(module)s] %(funcName)s %(lineno)s: %(levelname)-8s [%(process)d] %(message)s"
formatter = logging.Formatter(format_str)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logger.info("Environment Variables:")
for name, value in os.environ.items():
    logger.info(f"{name}: {value}")


class SupportedTask:
    """Supported tasks by text-generation-inference."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"


LOAD_BALANCING_PORT = 50050
MAX_TOKENS = 1024
TORCH_DIST_PORT = 29501
REPLICA_NUM = int(os.getenv("WORKER_COUNT", 1))
DEVICE_COUNT = torch.cuda.device_count()
TENSOR_PARALLEL = int(DEVICE_COUNT / REPLICA_NUM)
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR", "")
MODEL_PATH = "mlflow_model_folder/data/model"
MLMODEL_PATH = "mlflow_model_folder/MLmodel"
MODEL_ID = os.environ.get("MODEL_ID", MODEL_PATH)
task_type = SupportedTask.TEXT_GENERATION
default_generator_configs = {}

# AACS
aacs_threshold = int(os.environ.get("CONTENT_SAFETY_THRESHOLD", 2))
aacs_client = None


def init():
    """Initialize MII server and MII client."""
    global task_type
    global aacs_client

    # ACS setup
    try:
        logger.info("Setting up AACS")
        endpoint = os.environ.get("CONTENT_SAFETY_ENDPOINT", None)
        key = get_aacs_access_key()

        if not endpoint:
            raise Exception("CONTENT_SAFETY_ENDPOINT env not set for AACS.")
        if not key:
            raise Exception("CONTENT_SAFETY_KEY env not set for AACS.")

        # Create an Content Safety client
        headers_policy = HeadersPolicy()
        headers_policy.add_header("ms-azure-ai-sender", "llama")
        aacs_client = ContentSafetyClient(
            endpoint, AzureKeyCredential(key), headers_policy=headers_policy
        )
    except Exception as e:
        logger.error(f"AACS not configured. Bypassing content moderation. Error {e}")

    model_path = mii.utils.full_model_path(configs[mii.constants.MODEL_PATH_KEY])
    deployment_name = configs[mii.constants.DEPLOYMENT_NAME_KEY]
    model_name = configs[mii.constants.MODEL_NAME_KEY]
    task_name = configs[mii.constants.TASK_NAME_KEY]

    assert model_name is not None, "The model name should be set before calling init"
    assert task_name is not None, "The task name should be set before calling init"

    # Get task type of a model
    abs_mlmodel_path = os.path.join(MODEL_DIR, MLMODEL_PATH)
    mlmodel = {}
    if abs_mlmodel_path and os.path.exists(abs_mlmodel_path):
        with open(abs_mlmodel_path) as f:
            mlmodel = yaml.safe_load(f)

    check_model_flavors(mlmodel)

    try:
        start_server = True
        if int(os.getpid()) % configs.get("mii_configs").get("replica_num") != 0:
            start_server = False
            logger.info("Skip MII server setup for this process")

        if start_server:
            logger.info("Start server setup")
            mii.MIIServer(
                deployment_name,
                task_name,
                model_name,
                model_path,
                ds_optimize=configs[mii.constants.ENABLE_DEEPSPEED_KEY],
                ds_zero=configs[mii.constants.ENABLE_DEEPSPEED_ZERO_KEY],
                ds_config=configs[mii.constants.DEEPSPEED_CONFIG_KEY],
                mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                lb_config=configs.get(mii.constants.LOAD_BALANCER_CONFIG_KEY, None)
            )
            logger.info("Completed server setup")
            time.sleep(20)

            # run nvidia-smi
            logger.info("###### GPU INFO ######")
            logger.info(os.system("nvidia-smi"))
            logger.info("###### GPU INFO ######")
    except Exception as e:
        logger.error(f"MIIServer setup failed. Error {e}")
        raise e

    logger.info("Start client setup")

    global model
    model = None

    # In AML deployments both the GRPC client and server are used in the same process
    try:
        model = mii.MIIClient(task_name, "localhost", configs.get("mii_configs").get("port_number"))
    except Exception as e:
        logger.warning(f"MIIClient setup failed. Error {e}")
    logger.info("Completed client setup")


def run(data):
    """Call the model to get the text generation or chat completion results."""
    global model
    global task_type

    assert model is not None, "grpc client has not been setup when this model was created"

    try:
        # Check input content safety
        data, severity = get_safe_input(data)
        if severity > aacs_threshold:
            logger.warning(
                f"Input severity ({severity}) greater than aacs threshold ({aacs_threshold})."
            )
            return {}

        query, params = get_request_data(data)
        params = get_generator_params(params)
        logger.info(
            f"generating response for input_string: {query}, parameters: {params}"
        )

        response = model.query(query, **params)
        result_dict = {}
        if task_type == SupportedTask.CHAT_COMPLETION:
            result_dict = {"output": f"{response.response[0]}"}
        else:
            for i in range(len(response.response)):
                result_dict[str(i)] = response.response[i]
        logger.info(result_dict)
        time_taken = response.time_taken
        logger.info(f"time_taken: {time_taken}")
        return get_safe_response(result_dict)

    except Exception as e:
        return json.dumps({"error": str(e)})


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
    """
    Cs chunking utils.
    """

    def __init__(self, chunking_n=1000, delimiter="."):
        """Init function."""
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        """Chunk strings in a given length."""
        return (string[0 + i: length + i] for i in range(0, len(string), length))

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
        response = await loop.run_in_executor(executor, client.analyze_text, request)
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
        logger.info("SelfHarm severity: {}".format(response.self_harm_result.severity))
        severity = max(severity, response.self_harm_result.severity)
    if response.sexual_result is not None:
        logger.info("Sexual severity: {}".format(response.sexual_result.severity))
        severity = max(severity, response.sexual_result.severity)
    if response.violence_result is not None:
        logger.info("Violence severity: {}".format(response.violence_result.severity))
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
        return jsonable_result, 0

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
            "Cannot get AACS access key, both UAI_CLIENT_ID and CONTENT_SAFETY_KEY are not set, exiting..."
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


def check_model_flavors(mlmodel):
    """Check model task type and get default config."""
    global default_generator_configs
    global task_type
    if mlmodel:
        flavors = mlmodel.get("flavors", {})
        if "hftransformersv2" in flavors:
            task_type = flavors["hftransformersv2"]["task_type"]
            model_generator_configs = flavors["hftransformersv2"].get(
                "generator_config", {}
            )
            logger.info(f"model_generator_configs: {model_generator_configs}")
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
                f"updated default_generator_configs: {default_generator_configs}"
            )


def get_generator_params(params: dict):
    """Return accumulated generator params."""
    global default_generator_configs

    updated_params = {}
    updated_params.update(default_generator_configs)
    updated_params.update(params)
    # updated_params = add_and_validate_gen_params(updated_params, params)
    return updated_params


def get_request_data(
    request_string
) -> Tuple[Union[str, List[str]], Dict[str, Any]]:
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
            input_data = get_processed_input_data_for_chat_completion(input_data)

        input_data_formatted = {"query": input_data}
        logger.info(f"input_data_formatted: {input_data_formatted}")

        return input_data_formatted, params
    except Exception as e:
        raise Exception(
            json.dumps({
                "error": (
                    "Expected input format: \n"
                    '{"input_data": {"input_string": "<query>", '
                    '"parameters": {"k1":"v1", "k2":"v2"}}}.\n '
                    "<query> should be in below format:\n "
                    'For text-generation: ["str1", "str2", ...]\n'
                    'For chat-completion : [{"role": "user", "content": "str1"}, '
                    '{"role": "assistant", "content": "str2"} ....]'
                ),
                "exception": str(e),
            })
        )


def get_processed_input_data_for_chat_completion(data: List[str]) -> str:
    r"""Process chat completion input request.

    Taken from:
    https://github.com/facebookresearch/llama/blob/main/llama/generation.py

    example input:
    [
        {"role": "user", "content": "What is the tallest building in the world?"},
        {"role": "assistant", "content": "As of 2021, the Burj Khalifa in Dubai"},
        {"role": "user", "content": "and in Africa?"},
    ]
    example output:
    "[INST]What is the tallest building in the world?[/INST]
    As of 2021, the Burj Khalifa in Dubai\n
    [INST]and in Africa?[/INST]"
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible,
    while being safe.  Your answers should not include any harmful, unethical, racist, sexist,
    toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased
    and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead
    of answering something not correct. If you don't know the answer to a question, please
    don't share false information."""

    dialog = data
    history = ""
    assert len(dialog) > 0
    # add a system prompt if the first message is not a system prompt
    if dialog[0]["role"] != "system":
        dialog = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + dialog
    # add a tag to the system prompt
    dialog = [
        {
            "role": dialog[0]["role"],
            "content": B_SYS + dialog[0]["content"] + E_SYS
            # + dialog[1]["content"],
        }
    ] + dialog[1:]
    assert all([msg["role"] == "user" for msg in dialog[1::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[2::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    history += dialog[0]["content"]
    assert dialog[-1]["role"] == "user"
    for i, conv in enumerate(dialog[1:]):
        if i % 2 == 1:
            assert conv["role"] == "assistant"
            history += f"{conv['content'].strip()}"
        else:
            assert conv["role"] == "user"
            history += B_INST + f"{conv['content'].strip()}" + E_INST
    return history


def _allocate_processes(hostfile_path):
    from mii.deployment import _allocate_processes
    if hostfile_path is None:
        import tempfile
        hostfile_path = tempfile.NamedTemporaryFile(delete=False).name
        logger.info(f"hostfile_path: {hostfile_path}")
        num_gpu = DEVICE_COUNT
        with open(hostfile_path, "w") as f:
            f.write(f"localhost slots={num_gpu}")
    return _allocate_processes(hostfile_path, TENSOR_PARALLEL, REPLICA_NUM)


def _generate_load_balancer_config():
    replica_pool = _allocate_processes(hostfile_path=None)
    replica_configs = [
        ReplicaConfig(
            hostname=hostname,
            tensor_parallel_ports=list(range(LOAD_BALANCING_PORT+i*TENSOR_PARALLEL+1,
                                             LOAD_BALANCING_PORT+i*TENSOR_PARALLEL+1+TENSOR_PARALLEL)),
            torch_dist_port=i+TORCH_DIST_PORT,
            gpu_indices=gpu_indices
        )
        for i, (hostname, gpu_indices) in enumerate(replica_pool)
    ]
    load_balancer_config = LoadBalancerConfig(port=LOAD_BALANCING_PORT, replica_configs=replica_configs)
    return load_balancer_config


load_balancer_config = _generate_load_balancer_config()
is_70b_model = "Llama-2-70b" in MODEL_DIR or "Llama-2-70b-chat" in MODEL_DIR
replace_with_kernel_inject = not is_70b_model
configs = {
    'deployment_name': 'llama-deployment',
    'ds_config': None,
    'ds_optimize': True,
    'ds_zero': False,
    'load_balancer_config': load_balancer_config,
    'mii_configs': {
        'checkpoint_dict': None,
        'deploy_rank': load_balancer_config.replica_configs[0].gpu_indices,
        'dtype': torch.float16,
        'enable_cuda_graph': False,
        'enable_restful_api': False,
        'hf_auth_token': None,
        'load_with_sys_mem': True,
        'max_tokens': MAX_TOKENS,
        'meta_tensor': False,
        'port_number': LOAD_BALANCING_PORT,
        'profile_model_time': False,
        'replace_with_kernel_inject': replace_with_kernel_inject,
        'replica_num': REPLICA_NUM,
        'skip_model_check': True,
        'tensor_parallel': TENSOR_PARALLEL,
        'torch_dist_port': TORCH_DIST_PORT,
        'trust_remote_code': False
    },
    'model_name': MODEL_DIR,
    'model_path': MODEL_PATH,
    'task_name': 'text-generation'
}

logger.info(f"MII configs: {configs}")
