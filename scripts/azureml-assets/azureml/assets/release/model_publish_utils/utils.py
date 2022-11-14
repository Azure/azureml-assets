# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model Utils Class."""

from pathlib import Path
from subprocess import PIPE, run, STDOUT
import sys
from azureml.assets.util import logger


class ModelUtils:
    """Download the Model from url at a given commit into the specified model directory."""

    def __init__(self, model_url: str, model_commit_hash: str, model_download_type: str, model_dir: Path):
        """Create the base object for Model."""
        self.model_url = model_url
        self.model_commit_hash = model_commit_hash
        self.model_download_type = model_download_type
        self.model_dir = model_dir

    def _run(self, cmd, cwd: Path = "./") -> int:
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
            logger.print(f"Failed with error {result.stdout}.")
        else:
            logger.print(f"Successfully executed! Output: \n{result.stdout}")
        return result.returncode

    def _download_git_model(self) -> bool:
        """Download the Model."""
        """
        Initialize LFS
        Clones the model from the URL into the Model Directory.
        Deletes the incomplete model artifact in case of failure.
        Sets the HEAD to the commit hash.
        Deletes the .git folder in the downloaded artifacts.
        """
        clone_cmd = f"git clone {self.model_url} {self.model_dir}"
        result = self._run(clone_cmd)
        if result != 0:
            return False
        if self.model_commit_hash:
            cmd = f"git reset --hard {self.model_commit_hash}"
            self._run(cmd, cwd=self.model_dir)
            self._run('mkdir model_dir', cwd=self.model_dir)
            self._run('mv $(ls -I .git -I model_dir) model_dir', cwd=self.model_dir)
        return True

    def download_model(self) -> bool:
        """Prepare the Download Environment."""
        if self.model_download_type == 'git':
            download_success = self._download_git_model()
            return download_success
        else:
            logger.print('Unsupported Model Download Method.')
        return False
