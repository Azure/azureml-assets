# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model Utils Class."""

import os
from pathlib import Path
import shutil
import stat
from subprocess import PIPE, run, STDOUT
import sys
from azureml.assets import PathType
from azureml.assets.util import logger


def _onerror(func, path, exc_info):
    """Error Handler for shutil rmtree."""
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


class ModelDownloadUtils:
    """Download the Model from url at a given commit into the specified model directory."""

    def _run(cmd, cwd: Path = "./") -> int:
        """Run the command and returns the result."""
        logger.print(cmd)
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
            logger.log_error(f"Failed with error {result.stdout}.")
        else:
            logger.print(f"Successfully executed! Output: \n{result.stdout}")
        return result.returncode

    def _download_git_model(model_url, model_dir) -> bool:
        """Download the Model."""
        """
        Clones the model from the git URL into the Model Directory.
        Deletes the incomplete model artifact in case of failure.
        Deletes the .git folder in the downloaded artifacts.
        """
        clone_cmd = f"git clone {model_url} {model_dir}"
        result = ModelDownloadUtils._run(clone_cmd)
        if result != 0:
            return False
        git_path = model_dir / ".git"
        shutil.rmtree(git_path, onerror=_onerror)
        return True

    def _download_azure_artifacts(model_url, model_dir) -> bool:
        """Download model files from blobstore."""
        commands = f"""
        azcopy login --service-principal --application-id $SP_CLIENT_ID --tenant-id $SP_TENANT_ID
        azcopy cp --recursive=true {model_url} {model_dir}
        """
        result = ModelDownloadUtils._run(commands)
        # TODO: Handle error case correctly, since azcopy exits with 0 exit code, even in case of error.
        if result != 0:
            logger.log_warning(f"Failed to download model files with URL: {model_url}")
            return False
        return True

    def download_model(model_download_type, model_url, model_dir) -> bool:
        """Prepare the Download Environment."""
        if model_download_type == PathType.GIT:
            return ModelDownloadUtils._download_git_model(model_url, model_dir)
        if model_download_type == PathType.AZURE:
            return ModelDownloadUtils._download_azure_artifacts(model_url, model_dir)
        else:
            logger.print("Unsupported Model Download Method.")
        return False
