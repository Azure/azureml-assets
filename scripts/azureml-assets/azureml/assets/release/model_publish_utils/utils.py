# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from subprocess import run


class ModelUtils:
    def __init__(self, model_url, model_commit_hash, model_dir):
        self.model_url = model_url
        self.model_commit_hash = model_commit_hash
        self.model_dir = model_dir

    def _run_command(self, command, cwd="./"):
        try:
            run(command, cwd=cwd, shell=True)
        except Exception as e:
            logging.log("Failed to run {command} due to error: " + e)

    def _delete_downloaded_artifacts(self):
        delete_cmd = f"rm -rf {self.model_dir}"
        self._run_command(delete_cmd)

    def download_model(self) -> None:
        lfs_cmd = "git lfs install"
        self._run_command(lfs_cmd)
        clone_cmd = f"git clone {self.model_url} {self.model_dir}"
        self._run_command(clone_cmd)
        if self.model_commit_hash:
            commit_cmd = f"git reset --hard {self.model_commit_hash}"
            self._run_command(commit_cmd, cwd={self.model_dir})
