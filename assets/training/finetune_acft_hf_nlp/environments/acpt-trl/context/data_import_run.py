# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry file for FTaaS run."""

import os
import subprocess
import logging
from pathlib import Path
import shutil
from typing import Optional, List
from dataclasses import dataclass, field, fields

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError, PathNotFound
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml._common._error_definition.azureml_error import AzureMLError


logger = get_logger_app(
    "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import")


COMPONENT_NAME = "run_data_import"
_COMPONENTS_SCRIPTS_REL_PATH = Path("entry_point", "ftaas", "data_import")


@dataclass
class ComponentInput:
    """Dataclass for Ftaas pipeline component inputs."""

    value: str
    allowed_max_length: int = 128


@dataclass
class ComponentStr:
    """Dataclass for Ftaas pipeline component inputs."""

    value: str
    choices: list
    allowed_max_length: int = 128


@dataclass
class FtaasPipelineInputsValidator:
    """Dataclass for Ftaas pipeline inputs validator.

    NOTE The default values entered for each parameter is a dummy value.
    The actual values for them will be read from env variables.
    """

    _AZUREML_CR_DATA_CAPABILITY_PATH = "AZUREML_CR_DATA_CAPABILITY_PATH"
    _AZUREML_PARAMETER_PREFIX = "AZUREML_PARAMETER_"
    _AZUREML_INPUT_PREFIXES = ["/mnt", "azureml:/"]

    train_file_path: ComponentInput = field(
        default=ComponentInput("train.jsonl"),
        metadata={
            "help": "Path to training data"
        }
    )
    validation_file_path: ComponentInput = field(
        default=ComponentInput("validation.jsonl"),
        metadata={
            "help": "Path to validation data"
        }
    )

    def _validate_fields(self):
        """Validate field parameters for their types."""
        for param in fields(self):
            logger.info(f"Validating input: {param.name}")
            param_value = getattr(self, param.name)
            if isinstance(param_value, ComponentStr):
                self._str_param_validator(param.name)
            elif isinstance(param_value, int):
                self._int_param_validator(param.name)
            elif isinstance(param_value, float):
                self._float_param_validator(param.name)
            elif isinstance(param_value, ComponentInput):
                self._component_input_validator(param.name)

    def _str_param_validator(self, param_name: str):
        """Validate a string field."""
        env_var_name = self._AZUREML_PARAMETER_PREFIX + param_name
        user_passed_value = os.environ.get(env_var_name, None)
        if user_passed_value is None:
            logger.warning(f"Couldn't validate the parameter: {param_name}")
            return

        param = getattr(self, param_name)
        if param.choices and user_passed_value not in param.choices:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"Invalid value set for {param_name}: {user_passed_value}, allowed values are {param.choices}"
                    )
                )
            )
        if len(user_passed_value) > param.allowed_max_length:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"Invalid value set for {param_name}: {user_passed_value}, \
                            exceeds allowed max_length limit (128)"
                    )
                )
            )
        if not isinstance(user_passed_value, str):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"Invalid value set for {param_name}: {user_passed_value}"
                    )
                )
            )

    def _int_param_validator(self, param_name: str):
        """Validate an int field."""
        env_var_name = self._AZUREML_PARAMETER_PREFIX + param_name
        user_passed_value = os.environ.get(env_var_name, None)
        if user_passed_value is not None:
            try:
                int(user_passed_value)
            except Exception:
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value set for {param_name}: {user_passed_value}"
                        )
                    )
                )
        else:
            logger.warning(f"Couldn't validate the parameter: {param_name}")

    def _float_param_validator(self, param_name: str):
        """Validate a float field."""
        env_var_name = self._AZUREML_PARAMETER_PREFIX + param_name
        user_passed_value = os.environ.get(env_var_name, None)
        if user_passed_value is not None:
            try:
                float(user_passed_value)
            except Exception:
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value set for {param_name}: {user_passed_value}"
                        )
                    )
                )
        else:
            logger.warning(f"Couldn't validate the parameter: {param_name}")

    def _component_input_validator(self, param_name: str):
        """Validate a string field."""
        data_capability_path = os.environ.get(self._AZUREML_CR_DATA_CAPABILITY_PATH, None)
        if data_capability_path is None:
            logger.warning(f"Couldn't validate the parameter: {param_name}")
            return

        user_passed_value = os.path.join(data_capability_path, f'INPUT_{param_name}')

        if (
            user_passed_value is not None and
            not any([user_passed_value.startswith(prefix)
                    for prefix in self._AZUREML_INPUT_PREFIXES])
        ):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"Possible prefixes for {param_name} are {self._AZUREML_INPUT_PREFIXES}. \
                            Found {user_passed_value}"
                    )
                )
            )

    def __post_init__(self):
        """Validate a string field."""
        self._validate_fields()


def _copy_components_scripts():
    """Copy the component scripts packaged with packaged to the cwd."""
    from distutils.sysconfig import get_python_lib
    site_pkgs_root = get_python_lib()

    component_scripts_path = Path(site_pkgs_root, _COMPONENTS_SCRIPTS_REL_PATH)
    if component_scripts_path.is_dir():
        dst_folder = Path(__file__).parent.resolve()
        logger.info(
            f"Copying files from {component_scripts_path} to {dst_folder}")
        shutil.copytree(component_scripts_path, dst_folder, dirs_exist_ok=True)
    else:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    f"{component_scripts_path} doesn't exist."
                )
            )
        )


def decode_param_from_env_var(param_name: str) -> Optional[str]:
    """Decode the parameter value from the environment variables."""
    return os.environ.get(f"AZUREML_PARAMETER_{param_name}", None)


def decode_input_from_env_var(param_name: str) -> Optional[str]:
    """Decode the part input from the environment variables.

    NOTE: This approach works for mount.
    """
    try:
        return os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], f'INPUT_{param_name}')
    except Exception:
        return None


def decode_output_from_env_var(param_name: str) -> Optional[str]:
    """Decode the part input from the environment variables."""
    try:
        return os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], f'{param_name}')
    except Exception:
        return None


def add_optional_param(cmd, param_name):
    """Add optional parameters."""
    param_val = decode_param_from_env_var(param_name)
    if param_val is not None:
        cmd += ["--" + param_name, param_val]


def add_train_validation_file_path_input(cmd, input_name):
    """Add train validation path inputs."""
    input_val = decode_input_from_env_var(input_name)
    if input_val and os.path.isdir(input_val):
        if not os.listdir(input_val):
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    PathNotFound,
                    pii_safe_message=(
                        f"Invalid file path: {input_val}."
                    )
                )
            )
        input_val = os.path.join(input_val, os.listdir(input_val)[0])
    if input_val is not None and os.path.exists(input_val):
        cmd += ["--" + input_name, input_val]


def _run_subprocess_cmd(cmd: List[str], component_name: str):
    """Run the subprocess command."""
    logger.info(f"Starting the command: {cmd}")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        logger.info(line.strip())

    # get the return code
    return_code = process.wait()
    if return_code != 0:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    f"{component_name} failed"
                )
            )
        )
    logger.info(f"{component_name} completed successfully")


def _initiate_run():
    """Run the Data Import script."""
    # Data Import
    task_name = decode_param_from_env_var("task_name")
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.data_import.data_import",
        "--task_name", task_name,
        "--output_dataset", decode_output_from_env_var("output_dataset")
    ]
    add_train_validation_file_path_input(cmd=cmd, input_name="train_file_path")
    add_train_validation_file_path_input(
        cmd=cmd, input_name="validation_file_path")
    add_optional_param(cmd, "user_column_names")
    logger.info(f"Starting the command: {cmd}")

    _run_subprocess_cmd(cmd=cmd, component_name="Data Import")


@swallow_all_exceptions(time_delay=60)
def run():
    """Run the main function."""
    # validate inputs
    FtaasPipelineInputsValidator()

    # copy the component scripts to cwd
    # _copy_components_scripts()

    # run the component script
    _initiate_run()


if __name__ == "__main__":
    task_name = decode_param_from_env_var("task_name")
    # set logger
    set_logging_parameters(
        task_type=task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    run()
