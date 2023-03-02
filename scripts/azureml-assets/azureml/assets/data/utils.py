# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Data Utils Class."""
from pathlib import Path
from azureml.assets import PathType
from azureml.assets.util import logger
from subprocess import PIPE, run, STDOUT
import sys

class DataDownloadUtils:
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
    
    def _download_azure_artifacts(data_uri, data_dir) -> bool:
        """Download data files from blobstore.

        :param data_url: Publicly readable blobstore URI of data files
        :type data_url: str
        :param data_dir: local directory to data model to
        :type: Path
        """
        try:
            download_cmd = f"azcopy cp --recursive=true {data_uri} {data_dir}"
            result = DataDownloadUtils._run(download_cmd)
            # TODO: Handle error case correctly, since azcopy exits with 0 exit code, even in case of error.
            # https://github.com/Azure/azureml-assets/issues/283
            if result:
                logger.log_error(f"Failed to download model files with URL: {data_uri}")
                return False
            return True
        except Exception as e:
            logger.log_error(e)
            return False

    def download_data(data_path_type: PathType, data_uri: str, data_dir: Path) -> bool:
        """Prepare the Download Environment.

        :param model_path_type: Model path type
        :type model_path_type: PathType
        :param model_uri: uri to model files
        :type model_uri: str
        :param model_dir: local folder to download model files too
        :type model_dir: Path
        """
        if data_path_type == PathType.AZUREBLOB:
            return DataDownloadUtils._download_azure_artifacts(data_uri, data_dir)
        else:
            logger.print("Unsupported Data Download Method.")
        return False