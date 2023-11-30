# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry file for FTaaS run."""

import os
import subprocess
import logging
from pathlib import Path
import shutil
from dataclasses import dataclass, field, fields
from typing import Optional, List

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml._common._error_definition.azureml_error import AzureMLError


logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.finetune")


COMPONENT_NAME = "run_finetune"
_COMPONENTS_SCRIPTS_REL_PATH = Path("entry_point", "ftaas", "finetune")
_ALLOWED_MAX_STRING_LENGTH = 128


@dataclass
class ComponentInput:
    value: str
    allowed_max_length: int = _ALLOWED_MAX_STRING_LENGTH


@dataclass
class ComponentStr:
    value: str
    choices: list
    allowed_max_length: int = _ALLOWED_MAX_STRING_LENGTH


@dataclass
class FtaasPipelineInputsValidator:
    """Dataclass for Ftaas pipeline inputs validator.

    NOTE The default values entered for each parameter is a dummy value.
    The actual values for them will be read from env variables.
    """
    _AZUREML_CR_DATA_CAPABILITY_PATH = "AZUREML_CR_DATA_CAPABILITY_PATH"
    _AZUREML_PARAMETER_PREFIX = "AZUREML_PARAMETER_"
    _AZUREML_INPUT_PREFIXES = ["/mnt", "azureml:/"]

    mlflow_model_path: ComponentInput = field(
        default=ComponentInput("dummy_model_path"),
        metadata={
            "help": "MLflow model asset path"
        }
    )
    text_key: ComponentStr = field(
        default=ComponentStr("text", choices=[]),
        metadata={
            "help": (
                "key for text in an example. format your data keeping in mind that"
                "text is concatenated with ground_truth while finetuning in the form - text + groundtruth."
                'for eg. "text"="knock knock\n", \
                    "ground_truth"="whos there"; will be treated as "knock knock\nwhos there'
            )
        }
    )
    ground_truth_key: ComponentStr = field(
        default=ComponentStr("ground_truth", choices=[]),
        metadata={
            "help": (
                "Key for ground_truth in an example."
                "we take separate column for ground_truth to enable use cases like summarization, translation, "
                "question_answering, etc. which can be repurposed in form of text-generation where both text "
                "and ground_truth are needed. This separation is useful for calculating metrics."
                'for eg. "text"="Summarize this dialog:\n{input_dialogue}\nSummary:\n \
                    ", "ground_truth"="{summary of the dialogue}'
            )
        }
    )
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
    num_train_epochs: int = field(
        default=1,
        metadata={
            "help": "Number of epochs to train the model."
        }
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "Number of epochs to train the model."
        }
    )
    learning_rate: float = field(
        default=0.00002,
        metadata={
            "help": "Start learning rate."
        }
    )
    registered_model_name: ComponentStr = field(
        default=ComponentStr("dummy", choices=[]),
        metadata={
            "help": "Name of the registered model."
        }
    )

    def _validate_fields(self):
        """Validate field parameters for their types."""
        for param in fields(self):
            logger.info(f"Validating input: {param.name}")
            if isinstance(param.type, ComponentStr):
                self._str_param_validator(param.name)
            elif isinstance(param.type, int):
                self._int_param_validator(param.name)
            elif isinstance(param.type, float):
                self._float_param_validator(param.name)
            elif isinstance(param.type, ComponentInput):
                self._component_input_validator(param.name)

    def _str_param_validator(self, param_name: str):
        """Validate a string field."""
        env_var_name = self._AZUREML_PARAMETER_PREFIX + param_name
        user_passed_value = os.environ.get(env_var_name, None)
        param = getattr(fields(self), param_name)
        if param.value not in param.choices:
            raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value set for {param_name}: {user_passed_value}, \
                                allowed values are {param.choices}"
                        )
                    )
                )
        if len(param.value) > param.allowed_max_length:
            raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value set for {param_name}: {user_passed_value}, \
                                exceeds allowed max_length limit (128)"
                        )
                    )
                )
        if user_passed_value is not None:
            if not isinstance(user_passed_value, str):
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

    def _int_param_validator(self, param_name: str):
        """Validate an int field"""
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
        """Validate a float field"""
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
        """Validate a string field"""

        user_passed_value = os.path.join(os.environ[self._AZUREML_CR_DATA_CAPABILITY_PATH], f'INPUT_{param_name}')

        param = getattr(fields(self), param_name)
        if len(param.value) > param.allowed_max_length:
            raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value set for {param_name}: {user_passed_value}, \
                                exceeds allowed max_length limit (128)"
                        )
                    )
                )

        if (
            user_passed_value is not None and
            not any([user_passed_value.startswith(prefix) for prefix in self._AZUREML_INPUT_PREFIXES])
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
        else:
            logger.warning(f"Couldn't validate the parameter: {param_name}")

    def __post_init__(self):
        self._validate_fields()


def _copy_components_scripts():
    """Copy the component scripts packaged with packaged to the cwd."""
    from distutils.sysconfig import get_python_lib
    site_pkgs_root = get_python_lib()

    component_scripts_path = Path(site_pkgs_root, _COMPONENTS_SCRIPTS_REL_PATH)
    if component_scripts_path.is_dir():
        dst_folder = Path(__file__).parent.resolve()
        logger.info(f"Copying files from {component_scripts_path} to {dst_folder}")
        shutil.copytree(component_scripts_path, dst_folder, dirs_exist_ok=True)
    else:
        ACFTValidationException._with_error(
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


def add_optional_param(cmd, component_param_name: str, argparse_param_name: Optional[str] = None):
    if argparse_param_name is None:
        argparse_param_name = component_param_name
    param_val = decode_param_from_env_var(component_param_name)
    if param_val is not None and param_val != "":
        cmd += ["--" + argparse_param_name, param_val]


def add_optional_input(cmd, input_name):
    input_val = decode_input_from_env_var(input_name)
    if input_val is not None and os.path.exists(input_val):
        cmd += ["--" + input_name, input_val]


def _run_subprocess_cmd(cmd: List[str], component_name: str):
    """Utility function to run the subprocess command."""
    logger.info(f"Starting the command: {cmd}")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
    """Runs the model selector, preprocess, finetune and registration script."""

    # model selector
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.model_selector",
        "--task_name", "TextGeneration",
        "--mlflow_model_path", decode_input_from_env_var("mlflow_model_path"),
        "--output_dir", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "model_selector_output")
    ]
    _run_subprocess_cmd(cmd, component_name="Model selector")

    # preprocess
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.preprocess",
        "--task_name", "TextGeneration",
        "--text_key", decode_param_from_env_var("text_key"),
        "--batch_size", decode_param_from_env_var("batch_size"),
        "--pad_to_max_length", decode_param_from_env_var("pad_to_max_length"),
        "--max_seq_length", decode_param_from_env_var("max_seq_length"),
        "--train_file_path", os.path.join(decode_input_from_env_var("dataset_input"), "train_input.jsonl"),
        "--test_file_path", os.path.join(decode_input_from_env_var("dataset_input"), "train_input.jsonl"),
        "--model_selector_output", 
        os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "model_selector_output"),
        "--output_dir", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "preprocess_output")
    ]
    # add optional param ground_truth_key
    add_optional_param(cmd, "ground_truth_key")
    # add optional input validation_file_path
    add_optional_input(cmd, "validation_file_path")
    _run_subprocess_cmd(cmd, component_name="Preprocess")

    # finetune
    cmd = [
        "python", "-m", "torch.distributed.launch",
        "--nproc_per_node", decode_param_from_env_var('number_of_gpu_to_use_finetuning'),
        "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.finetune",
        "--apply_lora", decode_param_from_env_var('apply_lora'),
        "--merge_lora_weights", decode_param_from_env_var('merge_lora_weights'),
        "--lora_alpha", decode_param_from_env_var('lora_alpha'),
        "--lora_r", decode_param_from_env_var('lora_r'),
        "--lora_dropout", decode_param_from_env_var('lora_dropout'),
        "--num_train_epochs", decode_param_from_env_var('num_train_epochs'),
        "--max_steps", decode_param_from_env_var('max_steps'),
        "--per_device_train_batch_size", decode_param_from_env_var('per_device_train_batch_size'),
        "--per_device_eval_batch_size", decode_param_from_env_var('per_device_eval_batch_size'),
        "--auto_find_batch_size", decode_param_from_env_var('auto_find_batch_size'),
        "--optim", decode_param_from_env_var('optim'),
        "--learning_rate", decode_param_from_env_var('learning_rate'),
        "--warmup_steps", decode_param_from_env_var('warmup_steps'),
        "--weight_decay", decode_param_from_env_var('weight_decay'),
        "--adam_beta1", decode_param_from_env_var('adam_beta1'),
        "--adam_beta2", decode_param_from_env_var('adam_beta2'),
        "--adam_epsilon", decode_param_from_env_var('adam_epsilon'),
        "--gradient_accumulation_steps", decode_param_from_env_var('gradient_accumulation_steps'),
        "--eval_accumulation_steps", decode_param_from_env_var('eval_accumulation_steps'),
        "--lr_scheduler_type", decode_param_from_env_var('lr_scheduler_type'),
        "--precision", decode_param_from_env_var('precision'),
        "--seed", decode_param_from_env_var('seed'),
        "--enable_full_determinism", decode_param_from_env_var('enable_full_determinism'),
        "--dataloader_num_workers", decode_param_from_env_var('dataloader_num_workers'),
        "--ignore_mismatched_sizes", decode_param_from_env_var('ignore_mismatched_sizes'),
        "--max_grad_norm", decode_param_from_env_var('max_grad_norm'),
        "--evaluation_strategy", decode_param_from_env_var('evaluation_strategy'),
        "--evaluation_steps_interval", decode_param_from_env_var('evaluation_steps_interval'),
        "--eval_steps", decode_param_from_env_var('eval_steps'),
        "--logging_strategy", decode_param_from_env_var('logging_strategy'),
        "--logging_steps", decode_param_from_env_var('logging_steps'),
        "--metric_for_best_model", decode_param_from_env_var('metric_for_best_model'),
        "--resume_from_checkpoint", decode_param_from_env_var('resume_from_checkpoint'),
        "--save_total_limit", decode_param_from_env_var('save_total_limit'),
        "--apply_early_stopping", decode_param_from_env_var('apply_early_stopping'),
        "--early_stopping_patience", decode_param_from_env_var('early_stopping_patience'),
        "--early_stopping_threshold", decode_param_from_env_var('early_stopping_threshold'),
        "--apply_ort", decode_param_from_env_var('apply_ort'),
        "--apply_deepspeed", decode_param_from_env_var('apply_deepspeed'),
        "--deepspeed_stage", decode_param_from_env_var('deepspeed_stage'),
        "--model_selector_output",
        os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "model_selector_output"),
        "--preprocess_output", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "preprocess_output"),
        "--system_properties", decode_param_from_env_var("system_properties"),
        "--pytorch_model_folder", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "pytorch_model_folder"),
        "--mlflow_model_folder", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "mlflow_model_folder"),
        "--output_model", decode_output_from_env_var('output_model')
    ]
    _run_subprocess_cmd(cmd, component_name="Finetune")

    # validate lora weights
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.validate_lora_weights",
        "--mlflow_model_path", os.path.join(os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'], "mlflow_model_folder"),
        "--lora_weights_path", os.path.join(
            os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'],
            "pytorch_model_folder",
            "peft_adapter_weights"
        ),
        "--tokenizer_path", os.path.join(decode_input_from_env_var("mlflow_model_path") or "", "data", "tokenizer")
    ]
    _run_subprocess_cmd(cmd, component_name="Validate LoRA Weights")

    # model registration
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.register_model",
        "--task_name", "TextGeneration",
        "--finetune_args_path", os.path.join(
            os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'],
            "pytorch_model_folder",
            "finetune_args.json"
        ),
        "--registration_details_folder", decode_output_from_env_var('output_model'),
        "--model_path", os.path.join(
            os.environ['AZUREML_CR_DATA_CAPABILITY_PATH'],
            "pytorch_model_folder",
            "peft_adapter_weights"
        ),
        "--convert_to_safetensors", "true",
    ]
    add_optional_param(cmd=cmd, component_param_name="registered_model_name", argparse_param_name="model_name")
    _run_subprocess_cmd(cmd, component_name="Register Model")


@swallow_all_exceptions(time_delay=60)
def run():
    # validate inputs
    FtaasPipelineInputsValidator()

    # copy the component scripts to cwd
    # _copy_components_scripts()

    # run the component script
    _initiate_run()


if __name__ == "__main__":
    # set logger
    set_logging_parameters(
        task_type="TextGeneration",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    run()
