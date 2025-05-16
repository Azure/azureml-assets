# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utils."""

import re
import os
import psutil
import shutil
import sys
import time
from argparse import Namespace
from azure.ai.ml import MLClient
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
from azureml.core.run import Run
from azureml.model.mgmt.utils.exceptions import ModelImportErrorStrings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from subprocess import PIPE, run, STDOUT
from typing import Any, Dict, List, Optional, Tuple
from azureml.model.mgmt.utils.logging_utils import get_logger
from huggingface_hub.hf_api import HfApi, ModelInfo


HF_ENDPOINT = "https://huggingface.co"

KV_COLON_SEP = ":"
KV_EQ_SEP = "="
ITEM_COMMA_SEP = ","
ITEM_SEMI_COLON_SEP = ";"
BUFFER_SPACE = 1048576  # 1024 * 1024 KB (1GB)

hf_api = HfApi(endpoint=HF_ENDPOINT)
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


def get_mlclient(registry_name: str = None):
    """Return ML Client."""
    has_msi_succeeded = False
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        has_msi_succeeded = True
    except Exception:
        # Fall back to AzureMLOnBehalfOfCredential in case ManagedIdentityCredential does not work
        has_msi_succeeded = False
        logger.warning("ManagedIdentityCredential was not found in the compute. "
                       "Falling back to AzureMLOnBehalfOfCredential")

    if not has_msi_succeeded:
        try:
            credential = AzureMLOnBehalfOfCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            message = ModelImportErrorStrings.USER_IDENTITY_MISSING_ERROR
            raise MlException(
                message=message, no_personal_data_message=message,
                error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT,
                error=ex
            )

    if registry_name is None:
        run = Run.get_context(allow_offline=False)
        ws = run.experiment.workspace
        return MLClient(
            credential=credential,
            subscription_id=ws._subscription_id,
            resource_group_name=ws._resource_group,
            workspace_name=ws._workspace_name,
        )
    logger.info(f"Creating MLClient with registry name {registry_name}")
    return MLClient(credential=credential, registry_name=registry_name)


@log_execution_time
def copy_files(
    src_dir: Path,
    destn_dir: Path,
    include_pattern_str: str = r"^.*$",
    exclude_pattern_str: Optional[str] = None
) -> None:
    """Copy files to destination directory based on regex pattern provided for file name."""
    src_dir = Path(src_dir)
    destn_dir = Path(destn_dir)
    if not src_dir.is_dir():
        raise Exception("src path provided should be a dir")

    os.makedirs(destn_dir, exist_ok=True)
    include_pattern = re.compile(include_pattern_str)
    exclude_pattern = None if not exclude_pattern_str else re.compile(exclude_pattern_str)
    for fname in os.listdir(src_dir):
        src_abs_path = (src_dir / fname).absolute()
        if src_abs_path.is_dir():
            # recursively copy files
            copy_files(
                src_abs_path,
                (destn_dir / fname).absolute(),
                include_pattern_str,
                exclude_pattern_str
            )
        elif include_pattern.match(fname) and not (exclude_pattern and exclude_pattern.match(fname)):
            shutil.copy(os.path.join(src_dir, fname), destn_dir)


@log_execution_time
def move_files(src_dir: Path, destn_dir: Path, include_pattern_str: str = r"^.*$", ignore_case: bool = False) -> None:
    """Move files to destination directory based on regex pattern provided for file name."""
    src_dir = Path(src_dir)
    destn_dir = Path(destn_dir)
    if not src_dir.is_dir():
        raise Exception("src path provided should be a dir")

    os.makedirs(destn_dir, exist_ok=True)
    include_pattern = re.compile(include_pattern_str)
    if ignore_case:
        include_pattern = re.compile(include_pattern_str, re.IGNORECASE)

    dstn_abs_path = os.path.abspath(destn_dir)
    for fname in os.listdir(src_dir):
        src_abs_path = os.path.abspath(os.path.join(src_dir, fname))
        # cause AttributeError: 'PosixPath' object has no attribute 'rstrip' with Path object in src dir
        if os.path.isdir(src_abs_path):
            # recursively move files
            move_files(
                Path(src_abs_path),
                Path(dstn_abs_path) / fname,
                include_pattern_str
            )
        elif include_pattern.match(fname):
            shutil.move(src_abs_path, dstn_abs_path)


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
        model_list: List[ModelInfo] = hf_api.list_models(model_name=model_id)
        for info in model_list:
            if model_id == info.modelId:
                return info
    except Exception as e:
        message = ModelImportErrorStrings.ERROR_FETCHING_HUGGING_FACE_MODEL_INFO
        raise MlException(
            message=message.format(model_id=model_id, error=""),
            no_personal_data_message=message.format(model_id=model_id, error=e),
            error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT,
            error=e
        )


def get_system_time_utc():
    """Return formatted system time in UTC."""
    return "{0:%Y-%m-%d %H:%M:%S}".format(datetime.utcnow())


def get_file_or_folder_size(path: Path, size: int = 0) -> int:
    """Return file or folder size in bytes."""
    if os.path.isfile(path):
        return os.stat(path).st_size
    for entry in os.scandir(path):
        size += get_file_or_folder_size(entry)
    return size


def round_size(size: int) -> str:
    """Round size."""
    CONST = 1024
    dim = ["B", "KB", "MB", "GB", "TB"]
    count = 0
    while size / CONST > 1:
        count += 1
        size /= CONST
    return f"{size:.2f} {dim[count]}"


def get_filesystem_available_space_in_kb(path: Path = "./"):
    """Return filesystem available size in KB.

    :return: Return available size on VM in KB
    :rtype: int
    """
    partition = psutil.disk_usage(path)
    available_size_bytes = partition.free
    return available_size_bytes / 1024


def get_git_lfs_blob_size_in_kb(git_dir: Path) -> int:
    """Return total LFS blob size in KB.

    :param git_dir: git directory containing LFS file pointers
    :type git_dir: Path
    :return: total LFS blob size in KB
    :rtype: int
    """
    try:
        # 1. Execute git lfs ls-files -s command to get size of all blobs recursively
        # 2. Filter size using awk utility and translate them to KB.
        #    Expectation is that each blob dimension would be in [KB, MB, GB]
        # 3. Perform sum and return total size in KB
        cmd = (
            f"cd {str(git_dir)} && "
            "git lfs ls-files -s | "
            "awk -F'[()]| ' '{size=$5; unit=$6; if(unit==\"GB\") "
            "size*=1024*1024; else if(unit==\"MB\") size*=1024; print size}' | "
            "awk '{sum += $1} END {printf(\"%d\", sum)}'"
        )
        exit_code, stdout = run_command(cmd)
        if exit_code:
            raise Exception(f"Failed in fetching lfs blobsize: {stdout}")
        logger.info(f"total size: {stdout} KB")
        return int(stdout)
    except Exception as e:
        message = ModelImportErrorStrings.CMD_EXECUTION_ERROR
        raise MlException(
            message=message, no_personal_data_message=message.format(error=e),
            error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.COMPONENT,
            error=e
        )


class MlflowMetaConstants:
    """Mlflow consants."""

    IS_FINETUNED_MODEL = "is_finetuned_model"
    IS_ACFT_MODEL = "is_acft_model"
    BASE_MODEL_NAME = "base_model_name"
    BASE_MODEL_TASK = "base_model_task"


def fetch_mlflow_acft_metadata(
        is_finetuned_model: bool = False,
        base_model_name: str = None,
        base_model_task: str = None) -> dict:
    """Fetch metadata to be dumped in MlFlow MlModel File.

    :param is_finetuned_model: whether the model is finetuned one or base model
    :type is_finetuned_model: bool
    :param is_acft_model: whether the model using acft packages
    :type is_acft_model: bool
    :param base_model_name: name of the model
    :type base_model_name: str
    :param base_model_task: name of the base model task
    :type base_model_task: str

    :return: metadata
    :rtype: dict
    """
    metadata = {
        MlflowMetaConstants.IS_FINETUNED_MODEL: is_finetuned_model,
        MlflowMetaConstants.IS_ACFT_MODEL: True,
        MlflowMetaConstants.BASE_MODEL_NAME: base_model_name,
        MlflowMetaConstants.BASE_MODEL_TASK: base_model_task
    }

    return metadata
