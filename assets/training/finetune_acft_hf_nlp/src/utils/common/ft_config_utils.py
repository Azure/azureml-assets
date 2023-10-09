# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing utility functions to for ft_config."""

from typing import Tuple, List, Dict, Any
from yaml import safe_load

from utils.constants import FinetuneConfigConstants


class FtConfig:

    def __init__(self, ft_config_path: str) -> None:
        self.ft_config_path = ft_config_path
        self.ft_config_json = self.load_ft_config()

    def load_ft_config(self) -> Dict[str, Any]:
        with open(self.ft_config_path, 'r') as rptr:
            ft_config_json = safe_load(rptr)
        return ft_config_json

    def fetch_environment_setup(self) -> Tuple[List[str], List[str]]:
        if FinetuneConfigConstants.ENVIRONMENT_KEY not in self.ft_config_json:
            return [], []

        return (
            self.ft_config_json[FinetuneConfigConstants.ENVIRONMENT_KEY].get("uninstalls", []),
            self.ft_config_json[FinetuneConfigConstants.ENVIRONMENT_KEY].get("installs", []),
        )
