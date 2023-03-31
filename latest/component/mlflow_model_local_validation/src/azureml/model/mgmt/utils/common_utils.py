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
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azureml.core.run import Run, _OfflineRun
from contextlib import contextmanager
from pathlib import Path
from subprocess import PIPE, run, STDOUT
from typing import Any, Dict, Tuple


def log_execution_time(func, logger=None):
    """Decorate method to log execution time."""
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__!r} executed in {(t2-t1):.4f}s")
        return result
    return wrap_func


def run_command(cmd: str, cwd: Path = "./") -> Tuple[int, str]:
    """Run the command and returns the result."""
    print(cmd)
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
        print(f"Failed with error {result.stdout}.")
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
