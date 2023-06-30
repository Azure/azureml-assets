# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utils."""

import re
import os
import shutil
import sys
import time
from argparse import Namespace
from azure.ai.ml import MLClient
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azureml.core.run import Run, _OfflineRun
from azureml.model.mgmt.utils.exceptions import HuggingFaceErrorInFetchingModelInfo
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, run, STDOUT
from typing import Any, Dict, List, Tuple
from azureml.model.mgmt.utils.logging_utils import get_logger
from huggingface_hub.hf_api import HfApi, ModelInfo, ModelFilter


HF_ENDPOINT = "https://huggingface.co"

KV_COLON_SEP = ":"
KV_EQ_SEP = "="
ITEM_COMMA_SEP = ","
ITEM_SEMI_COLON_SEP = ";"


logger = get_logger(__name__)


def log_execution_time(func):
    """Decorate method to log execution time."""

    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        logger.info(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def run_command(cmd: str, cwd: Path = "./") -> Tuple[int, str]:
    """Run the command and returns the result."""
    logger.info(cmd)
    result = run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=PIPE,
        stderr=STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
    )
    if result.returncode != 0:
        logger.warning(f"Failed with error {result.stdout}.")
    return result.returncode, result.stdout


@contextmanager
def switch_dir(path: Path) -> None:
    """Context manager to change directory to `path` and revert back to origin post performing a task."""
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)


def get_json_header(token: str) -> Dict:
    """Return Header for JSON data."""
    return {
        "Authorization": f"Bearer {token}",
        "content-type": "application/json",
    }


def get_mlclient(registry_name=None, use_default=False):
    """Return mlclient object."""
    if use_default:
        credential = DefaultAzureCredential()
    else:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)

    if registry_name:
        return MLClient(
            credential=credential,
            registry_name=registry_name,
        )

    run = Run.get_context()
    if not isinstance(run, _OfflineRun):
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    return MLClient.from_config(credential)


def copy_file_paths_to_destination(src_dir: Path, destn_dir: Path, regex: str) -> None:
    """Copy files to destination directory [Non-recursively] based on regex pattern provided."""
    if not Path(src_dir).is_dir():
        raise Exception("src path provided should be a dir")
    if Path(destn_dir).exists():
        raise Exception("destination dir should be empty")
    os.makedirs(destn_dir)
    pattern = re.compile(regex)
    for file_name in os.listdir(src_dir):
        if pattern.match(file_name):
            shutil.copy(os.path.join(src_dir, file_name), destn_dir)


def create_namespace_from_dict(var: Any):
    """Create Namespace from dict."""
    if isinstance(var, dict):
        var = Namespace(**var)
        for key, val in var.__dict__.items():
            var.__dict__[key] = create_namespace_from_dict(val)
    return var


def get_dict_from_comma_separated_str(dict_str: str, item_sep: str, kv_sep: str) -> Dict:
    """Create and return dictionary from string.

    :param dict_str: string to be parsed for creating dictionary
    :type dict_str: str
    :param item_sep: char separator used for item separation
    :type item_sep: str
    :param kv_sep: char separator used for key-value separation. Must be different from item separator
    :type kv_sep: str
    :return: Resultant dictionary
    :rtype: Dict
    """
    if not dict_str:
        return {}
    item_sep = item_sep.strip()
    kv_sep = kv_sep.strip()
    if len(item_sep) > 1 or len(kv_sep) > 1:
        raise Exception("Provide single char as separator")
    if item_sep == kv_sep:
        raise Exception("item_sep and kv_sep are equal.")
    parsed_dict = {}
    kv_pairs = dict_str.split(item_sep)
    for item in kv_pairs:
        split = item.split(kv_sep)
        if len(split) == 2:
            key = split[0].strip()
            val = split[1].strip()
            # basic boolean conv. of input value
            if val.lower() == "true" or val.lower() == "false":
                val = bool(val.lower())
            parsed_dict[key] = val
    logger.info(f"get_dict_from_comma_separated_str: {dict_str} => {parsed_dict}")
    return parsed_dict


def get_list_from_comma_separated_str(list_str: str, item_sep: str) -> List:
    """Create and return list from string separted by `item_sep`.

    :param dict_str: string to be parsed for creating list
    :type dict_str: str
    :param item_sep: char used for item separation
    :type item_sep: str
    :return: Resultant list
    :rtype: List
    """
    if not list_str:
        return None
    return [x.strip() for x in list_str.split(item_sep) if x]


def retry(times):
    """Retry Decorator.

    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    """

    def decorator(func):
        def newfn(*args, **kwargs):
            attempt = 1
            while attempt <= times:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempt += 1
                    ex_msg = "Exception thrown when attempting to run {}, attempt {} of {}".format(
                        func.__name__, attempt, times
                    )
                    logger.warning(ex_msg)
                    if attempt == times:
                        logger.warning("Retried {} times when calling {}, now giving up!".format(times, func.__name__))
                        raise

        return newfn

    return decorator


@retry(3)
def fetch_huggingface_model_info(model_id) -> ModelInfo:
    """Return Hugging face model info."""
    try:
        model_list: List[ModelInfo] = HfApi(endpoint=HF_ENDPOINT).list_models(filter=ModelFilter(model_name=model_id))
        for info in model_list:
            if model_id == info.modelId:
                return info
    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(HuggingFaceErrorInFetchingModelInfo, model_id=model_id, error=e)
        )
