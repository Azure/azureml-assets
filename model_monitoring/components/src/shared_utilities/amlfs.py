# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data from the Azure Machine Learning File System."""

import json
import os
from typing import List
import uuid
from azureml.fsspec import AzureMachineLearningFileSystem
from shared_utilities.io_utils import np_encoder


def amlfs_get_as_json(remote_path: str) -> dict:
    """Read a json file located on the Azure Machine Learning file system.

    Args:
        remote_path: The path to the file.

    Returns:
        A dictionary representation of the json file.
    """
    fs = AzureMachineLearningFileSystem(remote_path)
    local_path = str(uuid.uuid4())
    fs.get(rpath=remote_path, lpath=local_path)
    with open(os.path.join(local_path, os.path.basename(remote_path)), "r") as fp:
        return json.loads(fp.read())


def amlfs_put_as_json(payload: dict, remote_path: str, filename: str):
    """Upload a dictionary object in json format to the Azure Machine Learning File System.

    Args:
        payload: The dictionary object to be uploaded.
        remote_path: The remote path on the Azure Machine Learning file system.
        filename: The name of the file.
    """
    content = json.dumps(payload, indent=4, default=np_encoder)

    local_path = os.path.join(str(uuid.uuid4()), filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "w") as fp:
        fp.write(content)

    fs = AzureMachineLearningFileSystem(remote_path)
    fs.put(lpath=local_path, rpath=remote_path)


def amlfs_glob(remote_glob_path: str) -> List[str]:
    """List the all of the files on the Azure Machine Learning file system which matches the glob pattern.

    Args:
        remote_glob_path: The remote glob path.

    Returns:
        The list of files which match the glob pattern.
    """
    fs = AzureMachineLearningFileSystem(remote_glob_path)
    return fs.glob(remote_glob_path)


def amlfs_download(remote_path: str, local_path: str):
    """Copy a file from one location to another on the Azure Machine Learning file system.

    Args:
        source_remote_path: The source path.
        target_remote_path: The target path.
    """
    fs = AzureMachineLearningFileSystem(remote_path)
    fs.get(rpath=remote_path, lpath=local_path, recursive=True)


def amlfs_upload(local_path: str, remote_path: str):
    """Copy a file from one location to another on the Azure Machine Learning file system.

    Args:
        source_remote_path: The source path.
        target_remote_path: The target path.
    """
    fs = AzureMachineLearningFileSystem(remote_path)
    fs.put(lpath=local_path, rpath=remote_path, recursive=True)
