# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides a decorator to log the execution time of functions."""

import os
import time
from logging_config import configure_logger

logger = configure_logger(__name__)


def log_execution_time(func):
    """
    Decorate a function to log the execution time.

    :param func: The function to be decorated.
    :return: The decorated function.
    """

    def wrapper(*args, **kwargs):
        """
        Calculate and log the execution time.

        :param args: Positional arguments for the decorated function.
        :param kwargs: Keyword arguments for the decorated function.
        :return: The result of the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if func.__name__ == "wait_until_server_healthy" and \
           os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
            logger.info(
                f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute."
            )
        return result

    return wrapper


def box_logger(message: str):
    """Log a message, but in a box."""
    row = len(message)
    h = ''.join(['+'] + ['-' * row] + ['+'])
    result = '\n' + h + '\n' + message + '\n' + h
    logger.info(result)


def map_env_vars_to_vllm_server_args() -> dict:
    """Map environment variables to VLLM server arguments."""
    env_to_cli_map = {
        "MODEL": "model",
        "TOKENIZER": "tokenizer",
        "REVISION": "revision",
        "TOKENIZER_REVISION": "tokenizer-revision",
        "TOKENIZER_MODE": "tokenizer-mode",
        "TRUST_REMOTE_CODE": "trust-remote-code",
        "DOWNLOAD_DIR": "download-dir",
        "LOAD_FORMAT": "load-format",
        "DTYPE": "dtype",
        "MAX_MODEL_LEN": "max-model-len",
        "WORKER_USE_RAY": "worker-use-ray",
        "PIPELINE_PARALLEL_SIZE": "pipeline-parallel-size",
        "TENSOR_PARALLEL_SIZE": "tensor-parallel-size",
        "BLOCK_SIZE": "block-size",
        "SEED": "seed",
        "SWAP_SPACE": "swap-space",
        "GPU_MEMORY_UTILIZATION": "gpu-memory-utilization",
        "MAX_NUM_BATCHED_TOKENS": "max-num-batched-tokens",
        "MAX_NUM_SEQS": "max-num-seqs",
        "MAX_PADDINGS": "max-paddings",
        "DISABLE_LOG_STATS": "disable-log-stats",
        "QUANTIZATION": "quantization",
        "ENGINE_USE_RAY": "engine-use-ray",
        "DISABLE_LOG_REQUESTS": "disable-log-requests",
        "MAX_LOG_LEN": "max-log-len"
    }

    cli_args = {}
    for env_var, cli_arg in env_to_cli_map.items():
        if env_var in os.environ:
            cli_args[cli_arg] = os.environ[env_var]

    box_logger(f"vLLM server arguments: {cli_args}")

    return cli_args
