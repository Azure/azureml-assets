# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""API server for foundation model inference.

This module implements a FastAPI-based server for serving foundation models with support for
OpenAI-compatible endpoints, content safety validation, and downstream engine proxying.
"""
# flake8: noqa

import torch
import httpx
import argparse
import copy
import json
import time
import uvicorn
import importlib.resources as impresources
import copy
import json
import os
import requests
import argparse
from typing import Any, Dict, Generator, Dict, Union

from torch.multiprocessing import set_start_method
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from importlib import resources as impresources
from contextlib import asynccontextmanager

from foundation.model.serve.aacs_handler import AACSValidator
from foundation.model.serve.fm_score import FMScore
from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve import api_server_setup
from foundation.model.serve.api_server_setup.protocol import CompletionResponse, CompletionRequest, ChatCompletionRequest, ChatCompletionResponse, CompletionStreamResponse, ChatCompletionStreamResponse
from foundation.model.serve.request_adapter import get_adapter
from foundation.model.serve.error_handler import to_azure_error_json_response
from foundation.model.serve.constants import EnvironmentVariables, TaskType, OpenAIEndpoints, CommonConstants
from foundation.model.serve.managed_inference import MIRPayload
from foundation.model.serve.replica_manager import InferenceResult

logger = configure_logger(__name__)

g_fmscorer: FMScore = None
DEFAULT_AACS_INFERENCE_URIS = [
    OpenAIEndpoints.V1_CHAT_COMLETIONS, OpenAIEndpoints.V1_COMPLETIONS]

g_aacs_threshold = int(os.environ.get(
    EnvironmentVariables.CONTENT_SAFETY_THRESHOLD, CommonConstants.CONTENT_SAFETY_THERESHOLD_DEFAULT))
g_aacs_client = None

TIMEOUT_KEEP_ALIVE = 30

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(openapi_url="/swagger.json", docs_url="/swagger.json")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Custom OpenAPI
# --------------------------------------------------


def filter_swagger_paths_by_tag(schema, tag):
    """Filter Swagger API paths by tag.
    
    Args:
        schema: The OpenAPI schema dictionary.
        tag: The tag to filter by.
        
    Returns:
        dict: The filtered schema containing only paths with the specified tag.
    """
    schema_copy = copy.deepcopy(schema)
    for path in list(schema["paths"].keys()):
        for op in list(schema["paths"][path].keys()):
            if "tags" in schema["paths"][path][op].keys() \
                    and tag not in schema["paths"][path][op]["tags"]:
                del schema_copy["paths"][path][op]
        if not schema_copy["paths"].get(path):
            del schema_copy["paths"][path]
    return schema_copy


def custom_openapi():
    """Populate OpenAPI schema values for Swagger UI.
    
    Returns:
        dict: The custom OpenAPI schema.
    """
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="AzureML Foundation Model Inference Server",
        version="1.0.0",
        summary="A server for inferencing AzureML Foundation Models.",
        routes=app.routes,
    )

    # Merge in custom /score schema
    swagger_file = (impresources.files(api_server_setup) / 'openapi.json')
    with open(swagger_file) as f:
        d = json.load(f)
    openapi_schema["paths"]["/score"] = d["/score"]

    # Determine which tag to filter on
    tag = ""
    app.openapi_schema = filter_swagger_paths_by_tag(openapi_schema, tag)
    return app.openapi_schema


app.openapi = custom_openapi


def _init_cuda_visible_devices():
    """Initialize CUDA visible devices environment variable.
    
    Sets CUDA_VISIBLE_DEVICES based on NVIDIA_VISIBLE_DEVICES or available GPUs.
    """
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

# --------------------------------------------------
# Proxy Middleware (Catch-all downstream forwarding)
# --------------------------------------------------


class ProxyMiddleware(BaseHTTPMiddleware):
    """Proxy middleware for forwarding unmatched requests to downstream engine.
    
    This middleware catches 404 responses from FastAPI and forwards them to
    the downstream inference engine, enabling pass-through of engine-specific endpoints.
    """

    async def dispatch(self, request: Request, call_next):
        """Dispatch HTTP requests, forwarding 404s to the downstream engine.
        
        Args:
            request: The incoming HTTP request.
            call_next: Callable to invoke the next middleware in the chain.
            
        Returns:
            Response: Either the FastAPI response or proxied downstream response.
        """
        # Try to process the request normally first
        response = await call_next(request)

        # If FastAPI found a matching route -> return its response
        if response.status_code != 404:
            return response

        path = request.url.path.lstrip("/")
        url = f"{get_downstream_url()}/{path}"

        aacs_inference_uri_list = DEFAULT_AACS_INFERENCE_URIS.copy()
        aacs_inference_uri = os.getenv(EnvironmentVariables.AACS_INFERENCE_URI)

        if aacs_inference_uri:
            aacs_inference_uri_list.extend(aacs_inference_uri.split(","))
        if path in aacs_inference_uri_list:
            try:
                request_json = await request.json()
                logger.info(f"Running aacs validation for uri {path}")
                _, severity = g_aacs_client.get_safe_input(request_json)
                if severity > g_aacs_threshold:
                    logger.warning(
                        f"Input severity ({severity}) greater than aacs threshold " f"({g_aacs_threshold}).",
                    )
                    return to_azure_error_json_response(
                        message="request input violates azure ai content safety setting.",
                        status_code=400,
                        headers={}
                    )
            except Exception as e:
                logger.error(
                    f"Error parsing the request for content safety {e}")
                raise

        safe_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() in ["content-type", "accept"]
        }

        client_host = request.client.host if request.client else "unknown"

        try:
            async with httpx.AsyncClient() as client:
                proxied = await client.request(
                    request.method,
                    url,
                    headers=safe_headers,
                    content=await request.body()
                )

            excluded_headers = ["server", "x-powered-by"]
            headers = {k: v for k, v in proxied.headers.items()
                       if k.lower() not in excluded_headers}

            logger.info(
                f"Proxied request from {client_host} "
                f"{request.method} {request.url.path} -> {url} "
                f"returned {proxied.status_code}"
            )

            return Response(
                content=proxied.content,
                status_code=proxied.status_code,
                headers=headers
            )
        except Exception as e:
            logger.error(f"Proxy error for {url}: {e}")
            return JSONResponse({"error": "Downstream unavailable"}, status_code=502)


def get_downstream_url():
    """Get the downstream engine URL.
    
    Returns:
        str: The URL of the downstream inference engine.
    """
    engine_startup_port = os.getenv(
        EnvironmentVariables.ENGINE_STARTUP_PORT, str(CommonConstants.DEFAULT_PORT))
    return f"http://{CommonConstants.HOST}:{engine_startup_port}"


def stream_response_to_generator(stream_response):
    """Convert streaming response to generator for SSE output.
    
    Args:
        stream_response: The streaming HTTP response from the downstream engine.
        
    Yields:
        str: Server-sent event formatted response chunks.
    """
    logger.info(f"Streaming response to generator")
    for line in stream_response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8') + '\n'
            if decoded_line.startswith("data: [DONE]"):
                logger.info(f"Streaming Response completed")
                yield f"data: [DONE]\n\n"
                break
            try:
                decoded_data = decoded_line.split("data: ")[1]
                decoded_json = json.loads(decoded_data)

                if task_type == TaskType.TEXT_GENERATION:
                    response_chunk = CompletionStreamResponse(**decoded_json)
                elif task_type == TaskType.CHAT_COMPLETION:
                    response_chunk = ChatCompletionStreamResponse(
                        **decoded_json)
                else:
                    logger.error(
                        f"Unsupported task type {task_type} for streaming response.")
                    raise Exception(
                        f"Unsupported task type {task_type} for streaming response.")
                response_chunk_json = response_chunk.model_dump_json(
                    exclude_unset=True)
                logger.info(f"Response chunk: {response_chunk_json}")
                yield f"data: {response_chunk_json}\n\n"
            except Exception as e:
                logger.error(
                    f"Error parsing decoded line: {decoded_line}, with exception: {e}")
                yield f"event: error\ndata: Error parsing decoded stream: {decoded_line}, with exception: {e}\n\n"
                yield "data: [DONE]\n\n"


# Register middleware
app.add_middleware(ProxyMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle input request validation errors.
    
    Args:
        request: The FastAPI request object.
        exc: The validation error exception.
        
    Returns:
        JSONResponse: Formatted error response with validation details.
    """
    return JSONResponse(
        status_code=422,
        content=(
            {
                "error": {
                    "code": "Invalid input",
                    "status": 422,
                    "message": "invalid input error",
                    "details": jsonable_encoder(exc.errors()),
                }
            }
        ),
    )


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions.
    
    Args:
        request: The FastAPI request object.
        exc: The exception.
        
    Returns:
        JSONResponse: Formatted error response.
    """
    msg = f"An error occurred during the request processing: {exc}"
    return to_azure_error_json_response(status_code=500, message=msg, headers={})


@app.get("/")
async def health():
    """Health check endpoint.
    
    Used by readiness probes to verify server health.
    
    Returns:
        str: Health status message.
    """
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
    return g_aacs_client.get_safe_response(result_dict)


class PrettyJSONResponse(JSONResponse):
    """Pretty-printed JSON response class."""

    def render(self, content: Any) -> bytes:
        """Render content as pretty-printed JSON.
        
        Args:
            content: The content to render.
            
        Returns:
            bytes: Pretty-printed JSON as bytes.
        """
        return json.dumps(content, ensure_ascii=False, indent=4).encode("utf-8")


# @app.get("/info", response_class=PrettyJSONResponse)
# async def info() -> Response:
#     """Model Metadata."""
#     return {
#         ModelInfo.MODEL_TYPE: g_model_info[ModelInfo.MODEL_TYPE],
#         ModelInfo.MODEL_PROVIDER: g_model_info[ModelInfo.MODEL_PROVIDER],
#         ModelInfo.MODEL_NAME: g_model_info[ModelInfo.MODEL_NAME]
#     }


@app.post("/completions", summary="Azure AI model inference: Text Generation", tags=TaskType.TEXT_GENERATION)
async def create_completion(
        request: CompletionRequest,
        raw_request: Request) -> CompletionResponse:
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    NOTE: Currently we do not support the following features:
        - function_call (Users should implement this by themselves)
        - logit_bias (to be supported by vLLM engine)
    """
    request = get_adapter(request, raw_request).adapt()
    response = send_openai_request(
        request, raw_request, OpenAIEndpoints.V1_COMPLETIONS)
    if response.status_code >= 400:
        return response

    if request.stream:
        generator = stream_response_to_generator(response)
        return StreamingResponse(content=generator,
                                 #  media_type="text/event-stream",
                                 media_type=response.headers["Content-Type"],
                                 headers=response.headers,)
    try:
        response_json = response.json()
        completion_response = CompletionResponse(**response_json)
        return g_aacs_client.get_safe_response(completion_response)
    except Exception as e:
        logger.error(f"Error parsing response as CompletionResponse: {e}")
        raise


@app.post("/chat/completions", summary="Azure AI model inference: Chat Completion")
async def create_chat_completion(
        request: ChatCompletionRequest,
        raw_request: Request) -> ChatCompletionResponse:
    """Completion API similar to OpenAI's API.

    See  https://platform.openai.com/docs/api-reference/chat/create
    for the API specification. This API mimics the OpenAI ChatCompletion API.

    """
    adapted_request = get_adapter(request, raw_request).adapt()
    response = send_openai_request(
        adapted_request, raw_request, OpenAIEndpoints.V1_CHAT_COMLETIONS)
    if response.status_code >= 400:
        return response

    if request.stream:
        generator = stream_response_to_generator(response)
        return StreamingResponse(content=generator,
                                 #  media_type="text/event-stream",
                                 media_type=response.headers["Content-Type"],
                                 headers=response.headers,)
    try:
        # temp hack to overwrite model to be the model_id thats being loaded
        response_json = response.json()
        # response_json["model"] = g_model_info[ModelInfo.MODEL_NAME]
        logger.debug(response_json)
        chat_completion_response = ChatCompletionResponse(**response_json)
        return g_aacs_client.get_safe_response(chat_completion_response)
    except Exception as e:
        logger.error(f"Error parsing response as ChatCompletionResponse: {e}")
        raise


def send_openai_request(request, raw_request, uri):
    """Send request to downstream engine with OpenAI-compatible schema.
    
    Args:
        request: The request object with OpenAI-compatible format.
        raw_request: The raw FastAPI request object.
        uri: The endpoint URI path.
        
    Returns:
        Response: The HTTP response from the downstream engine.
    """
    try:
        _, severity = g_aacs_client.get_safe_input(
            request.to_downstream_json())
        if severity > g_aacs_threshold:
            logger.warning(
                f"Input severity ({severity}) greater than aacs threshold " f"({g_aacs_threshold}).",
            )
            return to_azure_error_json_response(
                message="request input violates azure ai content safety setting.",
                status_code=400,
                headers={}
            )
    except Exception as e:
        logger.error(f"Error parsing the request for content safety {e}")
        raise

    # set model to be the correct model_id
    forward_headers = {}
    forward_headers['content-length'] = "application/json"
    payload = request.model_dump(exclude_none=True)          # Convert to Python dict
    
    response = requests.post(
        f"{get_downstream_url()}/{uri}", headers=forward_headers, json=payload)
    if response.status_code >= 400:
        try:
            return to_azure_error_json_response(
                status_code=response.status_code,
                message=response.json(),
                headers=response.headers,
            )
        except Exception:
            return to_azure_error_json_response(
                message=response.content,
                status_code=response.status_code,
                headers=response.headers,
            )

    return response


VLLM_SAMPLING_PARAMS = {
    "n": "Number of output sequences to return for the given prompt.",
    "best_of": "Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`.",
    "presence_penalty": "Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
    "frequency_penalty": "Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
    "temperature": "Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.",
    "top_p": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
    "top_k": "Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
    "use_beam_search": "Whether to use beam search instead of sampling.",
    "length_penalty": "Float that penalizes sequences based on their length. Used in beam search.",
    "early_stopping": 'Controls the stopping condition for beam search. Accepts `True`, `False`, or `"never"`.',
    "stop": "List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.",
    "stop_token_ids": "List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless they are special tokens.",
    "ignore_eos": "Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    "max_tokens": "Maximum number of tokens to generate per output sequence.",
    "logprobs": "Number of log probabilities to return per output token.",
    "skip_special_tokens": "Whether to skip special tokens in the output. Defaults to true.",
    "_batch_size": "Number of prompts to generate in parallel. Defaults to 1.",
}


def _normalize_generation_params(params: Dict) -> Dict:
    """Normalize generation-related parameter names and values for VLLM.
    
    Args:
        params: Dictionary of generation parameters.
        
    Returns:
        Dict: Normalized parameters compatible with VLLM.
    """
    # Map legacy or alternate keys
    for key in ("max_gen_len", "max_new_tokens"):
        if key in params:
            params["max_tokens"] = params[key]

    # Handle sampling and beam search configurations
    if not params.get("do_sample", True):
        logger.info("do_sample is false, setting temperature to 0.")
        params["temperature"] = 0.0

    if params.get("use_beam_search", False):
        logger.info("Beam search is enabled, setting temperature to 0.")
        params["temperature"] = 0.0
        params.setdefault("best_of", 2)
        if "best_of" not in params:
            logger.info("Beam search is enabled, setting best_of to 2.")

    # Handle EOS / Stop token mapping
    if "eos_token_id" in params:
        eos_token_id = params["eos_token_id"]
        params["stop_token_ids"] = [eos_token_id] if not isinstance(
            eos_token_id, list) else eos_token_id

    # Remove unsupported parameters
    unsupported_keys = set(params.keys()) - set(VLLM_SAMPLING_PARAMS.keys())
    for key in unsupported_keys:
        logger.warning(
            f"Parameter '{key}' is not supported by VLLM and will be removed.")
        params.pop(key, None)

    return params


def _get_payload_for_engine(data: Dict, task_type: str):
    """Prepare payload and endpoint path based on task type.
    
    Args:
        data: The input data dictionary.
        task_type: The task type (chat-completion or text-generation).
        
    Returns:
        tuple: A tuple containing (engine_payload, endpoint_uri).
        
    Raises:
        ValueError: If task type is not supported.
    """
    payload = MIRPayload.from_dict(data)
    logger.info(f"Processing new request with parameters: {payload.params}")

    params = _normalize_generation_params(payload.params)

    if task_type == TaskType.CHAT_COMPLETION:
        payload.convert_query_to_list()
        engine_payload = {"messages": payload.query, **params, "stream": False}
        endpoint = OpenAIEndpoints.V1_CHAT_COMLETIONS

    elif task_type == TaskType.TEXT_GENERATION:
        engine_payload = {"prompt": payload.query, **params, "stream": False}
        endpoint = OpenAIEndpoints.V1_COMPLETIONS

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return engine_payload, endpoint


async def send_request(data: Dict) -> (InferenceResult, Dict):
    """Execute inference request against VLLM engine.
    
    Args:
        data: The input data dictionary containing query and parameters.
        
    Returns:
        tuple: A tuple containing (InferenceResult, result_dict).
        
    Raises:
        Exception: If request processing fails.
    """
    global g_fmscorer, task_type

    try:
        data, severity = g_aacs_client.get_safe_input(data)
        if severity > g_aacs_threshold:
            logger.warning(
                f"Input severity ({severity}) greater than AACS threshold ({g_aacs_threshold})."
            )
            return {}

        task_type = os.getenv(
            EnvironmentVariables.TASK_TYPE, TaskType.CHAT_COMPLETION)
        payload, uri = _get_payload_for_engine(data, task_type)

        headers = {
            "content-type": "application/json",
            "User-Agent": "VLLMEngine Client",
        }

        logger.info(f"JSON payload for request execution: {payload}")

        start_time = time.time()
        response = requests.post(
            f"{get_downstream_url()}/{uri}", headers=headers, json=payload)
        end_time = time.time()

        generated_text = None
        if response.status_code == 200:
            output = response.json()
            if task_type == TaskType.CHAT_COMPLETION:
                generated_text = output["choices"][0]["message"]["content"]
            elif task_type == TaskType.TEXT_GENERATION:
                generated_text = output["choices"][0]["text"]

        results = (
            {"output": generated_text}
            if task_type == TaskType.CHAT_COMPLETION
            else {"output": [generated_text]}
        )

        return InferenceResult(generated_text), results

    except Exception as e:
        logger.exception(e)
        raise Exception(json.dumps({
            "error": "Error in processing request",
            "exception": str(e),
        }))


@asynccontextmanager
async def lifespan(app: FastAPI) -> Generator:
    """Initialize and shutdown events for each worker that is spawned in the main function.
    
    Args:
        app: The FastAPI application instance.
        
    Yields:
        None: Control is yielded during the application lifespan.
    """
    global g_aacs_client
    g_aacs_client = AACSValidator()
    AACS_error = g_aacs_client.aacs_setup()

    init_error = init_server(AACS_error)
    if init_error:
        logger.exception(init_error)
        raise init_error
    yield

app.router.lifespan_context = lifespan


def init_server(AACS_error: Union[None, Exception]):
    """Initialize text-generation-inference server and client.
    
    Args:
        AACS_error: Exception from AACS setup, if any.
        
    Returns:
        Exception or None: Exception if initialization fails, None otherwise.
    """
    global g_fmscorer

    try:

        _init_cuda_visible_devices()
        azureml_model_dir = os.getenv(EnvironmentVariables.AZUREML_MODEL_DIR)
        if azureml_model_dir is None:
            raise Exception(
                f"{EnvironmentVariables.AZUREML_MODEL_DIR} environment variable is not set.")
        g_fmscorer = FMScore()
        g_fmscorer.init()
        logger.info("Server started successfully")
        if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
            for k, v in os.environ.items():
                logger.info(f"env: {k} = {v}")
            if AACS_error:
                logger.warning(
                    f"AACS was not configured. Content moderation bypassed in setup. Error {AACS_error}")
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

    workers = int(os.getenv("WORKER_COUNT", 1))
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
