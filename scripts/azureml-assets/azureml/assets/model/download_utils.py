# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model Utils Class."""

import os
import subprocess
import shutil
import stat
import sys
from pathlib import Path
from azureml.assets.util import logger
from azure.ai.ml._azure_environments import (
    AzureEnvironments,
    _get_default_cloud_name,
    _get_storage_endpoint_from_metadata
)


def _onerror(func, path, exc_info):
    """Error Handler for shutil rmtree."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def run_cmd(cmd, cwd: Path = "./") -> int:
    """Run the command and returns the result."""
    logger.print(cmd)
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=sys.stdout.encoding,
        errors="ignore",
    )
    if result.returncode != 0:
        logger.log_error(f"Failed with error {result.stdout}.")
    else:
        logger.print(f"Successfully executed! Output: \n{result.stdout}")
    return result.returncode


def download_git_model(model_uri: str, model_dir: Path) -> bool:
    """Download model files from GIT repository.

    :param model_url: git clonable uri of a public repo
    :type model_url: str
    :param model_dir: local directory to clone model to
    :type: Path
    """
    clone_cmd = f"git clone {model_uri} {model_dir}"
    result = run_cmd(clone_cmd)
    if result != 0:
        return False
    git_path = model_dir / ".git"
    shutil.rmtree(git_path, onerror=_onerror)
    return True


def copy_azure_artifacts(src_uri: str, dstn_uri: str) -> bool:
    """Copy azure blobstorage artifacts to from blobstore.

    :param src_uri: a valid azure blobstorage source URI
    :type model_url: str
    :param dstn_uri: a valid azure blobstorage destination URI
    :type dstn_uri: str
    """
    try:
        download_cmd = f"azcopy copy '{src_uri}' '{dstn_uri}' " \
                       "--recursive --skip-version-check --output-level essential"

        # AzureCloud, USGov, and China clouds should all the trusted Microsoft
        # suffixes built into azcop by default. If the cloud is not one of these,
        # then we need to add the appropriate cloud-specific suffix ourselves.
        if _get_default_cloud_name() not in [AzureEnvironments.ENV_DEFAULT,
                                             AzureEnvironments.ENV_US_GOVERNMENT,
                                             AzureEnvironments.ENV_CHINA]:
            suffix = _get_storage_endpoint_from_metadata()
            if not suffix.startswith("."):
                suffix = "." + suffix
            download_cmd += f" --trusted-microsoft-suffixes {suffix}"

        result = run_cmd(download_cmd)
        logger.print(f"azcopy result {result}")
        # TODO: Handle error case correctly, since azcopy exits with 0 exit code, even in case of error.
        # https://github.com/Azure/azureml-assets/issues/283
        if result:
            logger.log_error(f"Failed to download model files with URL: {src_uri}")
            return False
        return True
    except Exception as e:
        logger.log_error(e)
        return False
