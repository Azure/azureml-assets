# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing Component Literals."""

from dataclasses import dataclass


@dataclass
class ComponentConstants:
    """Component constants."""
    MODEL_ASSET_ID_NOT_FOUND = "MODEL_ASSET_ID_NOT_FOUND"
    COMPONENT_ASSET_ID_NOT_FOUND = "COMPONENT_ASSET_ID_NOT_FOUND"


@dataclass
class EnvironmentConstants:
    """Environment to dynamically setup environment."""
    ENVIRONMENT_TMP_FOLDER = "AZUREML_ENVIRONMENT_TMP_FOLDER"
    UNINSTALL_REQUIREMENTS_PATH = "uninstall_requirements.txt"
    REQUIREMENTS_PATH = "requirements.txt"


@dataclass
class FinetuneConfigConstants:
    """Define the constants for finetune config here."""
    FINETUNE_CONFIG_PATH = "AZUREML_finetune_config.json"
    ENVIRONMENT_KEY = "environment"
