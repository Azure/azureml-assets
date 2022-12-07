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

    def _download_git_model(model_uri: str, model_dir: Path) -> bool:
        """Download model files from GIT repository.

        :param model_url: git clonable uri of a public repo
        :type model_url: str
        :param model_dir: local directory to clone model to
        :type: Path
        """
        clone_cmd = f"git clone {model_uri} {model_dir}"
        result = ModelDownloadUtils._run(clone_cmd)
        if result != 0:
            return False
        git_path = model_dir / ".git"
        shutil.rmtree(git_path, onerror=_onerror)
        return True

    def _download_azure_artifacts(model_uri, model_dir) -> bool:
        """Download model files from blobstore.

        :param model_url: Publicly readable blobstore URI of model files
        :type model_url: str
        :param model_dir: local directory to download model to
        :type: Path
        """
        try:
            download_cmd = f"azcopy cp --recursive=true {model_uri} {model_dir}"
            result = ModelDownloadUtils._run(download_cmd)
            # TODO: Handle error case correctly, since azcopy exits with 0 exit code, even in case of error.
            # https://github.com/Azure/azureml-assets/issues/283
            if result:
                logger.log_error(f"Failed to download model files with URL: {model_uri}")
                return False
            return True
        except Exception as e:
            logger.log_error(e)
            return False

    def download_model(model_path_type: PathType, model_uri: str, model_dir: Path) -> bool:
        """Prepare the Download Environment.

        :param model_path_type: Model path type
        :type model_path_type: PathType
        :param model_uri: uri to model files
        :type model_uri: str
        :param model_dir: local folder to download model files too
        :type model_dir: Path
        """
        if model_path_type == PathType.GIT:
            return ModelDownloadUtils._download_git_model(model_uri, model_dir)
        if model_path_type == PathType.AZUREBLOB:
            return ModelDownloadUtils._download_azure_artifacts(model_uri, model_dir)
        else:
            logger.print("Unsupported Model Download Method.")
        return False
