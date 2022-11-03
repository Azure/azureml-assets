# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from subprocess import run


class ModelUtils:
    def __init__(self, model_url, model_commit_hash, model_dir):
        self.model_url = model_url
        self.model_commit_hash = model_commit_hash
        self.model_dir = model_dir

    def download_model(self) -> None:
        cmd = f"git clone {self.model_url} {self.model_dir}"
        run(cmd, shell=True)
        if self.model_commit_hash:
            run(
                f"git reset --hard {self.model_commit_hash}",
                cwd={self.model_dir},
                shell=True,
            )
