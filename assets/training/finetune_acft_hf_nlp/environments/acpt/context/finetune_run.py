# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry file for FTaaS run."""

import os
import subprocess
import logging
import json
from pathlib import Path
import shutil
import time
import base64
from dataclasses import dataclass, field, fields
from typing import Optional, List

from azureml.acft.contrib.hf import VERSION, PROJECT_NAME
from azureml.acft.contrib.hf.nlp.constants.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS, SaveFileConstants
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.accelerator.utils.run_utils import is_main_process, get_process_name, wait_at_barrier
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals, SystemSettings
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.contrib.hf.nlp.constants.constants import Tasks


logger = get_logger_app("azureml.acft.contrib.hf.nlp.entry_point.finetune")


COMPONENT_NAME = "run_finetune"
_COMPONENTS_SCRIPTS_REL_PATH = Path("entry_point", "ftaas", "finetune")
_ALLOWED_MAX_STRING_LENGTH = 128
PEFT_ADAPTER_WEIGHTS_DIR = "peft_adapter_weights"
CHAT_KEY = "messages"
_AZUREML_FT_ENALBE_MULTI_NODE_SUPPORT = "_AZUREML_FT_ENALBE_MULTI_NODE_SUPPORT"

TASK_SPECIFIC_PARAMS = {
    "preprocess": {
        Tasks.TEXT_GENERATION: {
            "text_key": {"type": "param", "required": True},
            "ground_truth_key": {"type": "param", "required": False}
        },
        Tasks.CHAT_COMPLETION: {}
    },
    "validate_lora_weights": {
        Tasks.TEXT_GENERATION: {
            "text_or_chat_key": {"type": "param", "required": True, "env_name": "text_key"}
        },
        Tasks.CHAT_COMPLETION: {
            "text_or_chat_key": CHAT_KEY
        }
    }
}


def retry_with_backoff(delay: int = 1, retries: int = 3):
    """Retry with backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_retry = 0
            current_delay = delay
            while current_retry < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    current_retry += 1
                    if current_retry >= retries:
                        raise e
                    logger.warning(f"Failed to execute function '{func.__name__}'. \
                                   Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= 2
        return wrapper
    return decorator


@dataclass
class ComponentInput:
    """Dataclass for Ftaas pipeline component inputs."""

    value: str
    allowed_max_length: int = _ALLOWED_MAX_STRING_LENGTH


@dataclass
class ComponentStr:
    """Dataclass for Ftaas pipeline component inputs."""

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
        """Validate a string field."""
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
    """Add optional parameters."""
    if argparse_param_name is None:
        argparse_param_name = component_param_name
    param_val = decode_param_from_env_var(component_param_name)
    if param_val is not None and param_val != "":
        cmd += ["--" + argparse_param_name, param_val]


def add_optional_input(cmd, input_name):
    """Add optional inputs."""
    input_val = decode_input_from_env_var(input_name)
    if input_val is not None and os.path.exists(input_val):
        cmd += ["--" + input_name, input_val]


def add_task_specific_params(cmd: List[str], task_name: str, component_name: str):
    """Add task specific params based on task_name."""
    logger.info(f"Adding task_specific params for {task_name} task, for {component_name} component")

    # Get component specific params
    component_dict = TASK_SPECIFIC_PARAMS.get(component_name, {})
    # Get task specific params
    param_dict = component_dict.get(task_name, {})

    # Loop over params and add on basis of type and required
    for param in param_dict:
        param_val = param_dict.get(param)
        if isinstance(param_val, str):
            # If its already string, then add the param as it is
            # necessary to add params which are not directly available as env variables
            # eg. joining multiple inputs for getting path to a file
            cmd += ["--" + param, param_val]
        elif isinstance(param_val, dict):
            # If its dict, then parse the param and add to cmd
            component_param_name = param_val.get("env_name", param)
            if param_val.get("required", False) is True:
                if param_val["type"] == "param":
                    cmd += ["--" + param, decode_param_from_env_var(component_param_name)]
                elif param_val["type"] == "input":
                    cmd += ["--" + param, decode_input_from_env_var(component_param_name)]
            else:
                if param_val["type"] == "param":
                    add_optional_param(cmd, component_param_name, param)
                elif param_val["type"] == "input":
                    add_optional_input(cmd, param)


def _is_multi_node_enabled() -> bool:
    """
    To check if multi-node support is enabled.

    Multi-node support is enabled by setting the environment variable _AZUREML_FT_ENABLE_MULTI_NODE_SUPPORT to "true".
    If this environment variable is not defined, the function will return False by default.

    :return: True if multi-node support is enabled, False otherwise.
    """
    if os.environ.get(_AZUREML_FT_ENALBE_MULTI_NODE_SUPPORT, None) == "true":
        return True
    else:
        return False


def _run_subprocess_cmd(cmd: List[str], component_name: str, completion_files_folder: str,
                        single_run=True, number_of_processes=1):
    """Run the subprocess command."""
    logger.info(f"Starting the command: {cmd}")
    completion_file = Path(completion_files_folder, f"{component_name}.complete.txt")
    barrier_file = Path(completion_files_folder, f"{component_name}.barrier.txt")
    Path(completion_files_folder).mkdir(parents=True, exist_ok=True)
    if completion_file.exists():
        logger.info(f"Skipping {component_name} as completion file exists: {completion_file}")
        return
    process_name = get_process_name()
    if single_run and _is_multi_node_enabled():
        if not barrier_file.exists():
            Path(barrier_file).touch()
            logger.info(f'Barrier file {barrier_file} is created by process name {process_name}')
        if is_main_process():
            logger.info(f"Executing command: {cmd} in single run mode. Process name is {process_name}")
            # Not setting stdout and stderr will stream all the logs directly to stdout
            process = subprocess.Popen(cmd)

            # get the return code
            return_code = process.wait()
            if return_code != 0:
                intermediate_folder = decode_output_from_env_var("intermediate_folder")
                completion_files_folder = os.path.join(intermediate_folder, "completion_files")
                shutil.rmtree(completion_files_folder, ignore_errors=True)
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"{component_name} failed"
                        )
                    )
                )
            logger.info(f"{component_name} completed successfully")
            Path(completion_file).touch()
            logger.info(f"Created completion file: {completion_file}")
        logger.info(f"Waiting for completion file: {completion_file}, by rank : {process_name}")
        while not completion_file.exists():
            if is_main_process():
                time.sleep(1)
            else:
                pass
        logger.info(f"Process name on subprocess entering count barrier : {process_name}")
        wait_at_barrier(barrier_file, number_of_processes)
        logger.info(f"Process name on subprocess exiting count barrier : {process_name}")
    else:
        logger.info(f"Executing the command: {cmd}.")
        # Not setting stdout and stderr will stream all the logs directly to stdout
        process = subprocess.Popen(cmd)

        # get the return code
        return_code = process.wait()
        if return_code != 0:
            intermediate_folder = decode_output_from_env_var("intermediate_folder")
            completion_files_folder = os.path.join(intermediate_folder, "completion_files")
            shutil.rmtree(completion_files_folder, ignore_errors=True)
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"{component_name} failed"
                    )
                )
            )
        logger.info(f"{component_name} completed successfully")
        Path(completion_files_folder).mkdir(parents=True, exist_ok=True)
        Path(completion_file).touch()
        logger.info(f"Created completion file: {completion_file}")


def cleanup(completion_files_folder: str, model_selector_output: str,
            preprocess_output: str, pytorch_model_folder: str, mlflow_model_folder: str):
    """Cleanup intermediate files and folders that are not needed after successful run completion."""
    logger.info("Cleaning up intermediate files/folders...")
    try:
        shutil.rmtree(completion_files_folder, ignore_errors=True)
        shutil.rmtree(model_selector_output, ignore_errors=True)
        shutil.rmtree(preprocess_output, ignore_errors=True)
        shutil.rmtree(mlflow_model_folder, ignore_errors=True)
        if os.path.exists(pytorch_model_folder):
            for name in os.listdir(pytorch_model_folder):
                if name != PEFT_ADAPTER_WEIGHTS_DIR:  # don't preserve anything except peft adapter weights
                    path = os.path.join(pytorch_model_folder, name)
                    if os.path.isfile(path):
                        os.unlink(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path, ignore_errors=True)
        logger.info("Cleanup of intermediate files/folders completed successfully")
    except Exception as e:
        logger.error(f"Cleanup of intermediate files/folders failed. Ignoring error: {e}")


def initiate_run():
    """Run the model selector, preprocess, finetune and registration script. Cleanup after that."""
    # intermediate folder preserves files/folders that are needed across singularity preemptions
    intermediate_folder = decode_output_from_env_var("intermediate_folder")
    completion_files_folder = os.path.join(intermediate_folder, "completion_files")
    model_selector_output = os.path.join(intermediate_folder, "model_selector_output")
    preprocess_output = os.path.join(intermediate_folder, "preprocess_output")
    pytorch_model_folder = os.path.join(intermediate_folder, "pytorch_model_folder")
    mlflow_model_folder = os.path.join(intermediate_folder, "mlflow_model_folder")

    _initiate_run(completion_files_folder, model_selector_output,
                  preprocess_output, pytorch_model_folder, mlflow_model_folder)

    if is_main_process():
        cleanup(completion_files_folder, model_selector_output,
                preprocess_output, pytorch_model_folder, mlflow_model_folder)


def parse_to_int(s):
    """
    To parse string to integer with default value in case of failure as one.

    s: String which need to be parsed to integer.
    """
    try:
        return int(s)
    except ValueError:
        raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid value {s} entered, it should be a number."
                        )
                    )
                )


def parse_system_properties(arg_str: str):
    """Parse system properties."""
    if not arg_str:
        return {}

    try:
        json_bytes = base64.b64decode(arg_str)
        json_str = json_bytes.decode('utf-8')
        system_properties_dict = json.loads(json_str)
        return system_properties_dict
    except ValueError:
        logger.error(f"Failed to parse system properties: {arg_str}")
        return {}


def _initiate_run(completion_files_folder: str, model_selector_output: str,
                  preprocess_output: str, pytorch_model_folder: str, mlflow_model_folder: str):
    """Run the model selector, preprocess, finetune and registration script."""
    # get task name
    task_name = decode_param_from_env_var("task_name")
    num_nodes = parse_to_int(decode_param_from_env_var("Node_Count"))
    num_gpus = parse_to_int(decode_param_from_env_var("number_of_gpu_to_use_finetuning"))
    logger.info(f'Nodes are {num_nodes} , gpus are : {num_gpus}')

    # get system properties
    system_properties = parse_system_properties(decode_param_from_env_var("system_properties"))

    # set log_level_debug as environment parameter
    log_level_debug_enabled = \
        system_properties.get(SystemSettings.LOG_LEVEL_DEBUG, False) if system_properties else False
    os.environ[SystemSettings.LOG_LEVEL_DEBUG] = str(log_level_debug_enabled)

    # model selector
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.model_selector",
        "--task_name", task_name,
        "--output_dir", model_selector_output
    ]
    add_optional_input(cmd, "mlflow_model_path")
    add_optional_input(cmd, "pytorch_model_path")
    _run_subprocess_cmd(cmd, component_name="model_selector", completion_files_folder=completion_files_folder,
                        single_run=True, number_of_processes=num_gpus)
    # preprocess
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.preprocess",
        "--task_name", task_name,
        "--batch_size", decode_param_from_env_var("batch_size"),
        "--pad_to_max_length", decode_param_from_env_var("pad_to_max_length"),
        "--max_seq_length", decode_param_from_env_var("max_seq_length"),
        "--train_file_path", os.path.join(decode_input_from_env_var("dataset_input") or "", "train_input.jsonl"),
        "--test_file_path", os.path.join(decode_input_from_env_var("dataset_input") or "", "train_input.jsonl"),
        "--num_train_epochs", decode_param_from_env_var('num_train_epochs'),
        "--model_selector_output", model_selector_output,
        "--output_dir", preprocess_output
    ]
    # add task_specific params
    add_task_specific_params(cmd, task_name, component_name="preprocess")
    # add optional input validation_file_path
    validation_file_path = os.path.join(decode_input_from_env_var("dataset_input") or "", "validation_input.jsonl")
    if os.path.isfile(validation_file_path):
        cmd += ["--validation_file_path", validation_file_path]

    num_retries = system_properties.get("num_retries", 3) if system_properties else 3

    @retry_with_backoff(delay=2, retries=num_retries)
    def _run_preprocess_cmd_with_retries():
        _run_subprocess_cmd(cmd, component_name="preprocess", completion_files_folder=completion_files_folder,
                            single_run=True, number_of_processes=num_gpus)

    _run_preprocess_cmd_with_retries()

    # finetune
    if not _is_multi_node_enabled():
        cmd_base = ["python", "-m", "torch.distributed.launch", "--nproc_per_node",
                    decode_param_from_env_var('number_of_gpu_to_use_finetuning'), "-m"]
    else:
        cmd_base = ["python", "-m"]

    cmd = [
        "azureml.acft.contrib.hf.nlp.entry_point.finetune.finetune",
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
        "--save_strategy", decode_param_from_env_var('save_strategy'),
        "--save_steps", decode_param_from_env_var('save_steps'),
        "--save_total_limit", decode_param_from_env_var('save_total_limit'),
        "--apply_early_stopping", decode_param_from_env_var('apply_early_stopping'),
        "--early_stopping_patience", decode_param_from_env_var('early_stopping_patience'),
        "--early_stopping_threshold", decode_param_from_env_var('early_stopping_threshold'),
        "--apply_ort", decode_param_from_env_var('apply_ort'),
        "--apply_deepspeed", decode_param_from_env_var('apply_deepspeed'),
        "--deepspeed_stage", decode_param_from_env_var('deepspeed_stage'),
        "--model_selector_output", model_selector_output,
        "--preprocess_output", preprocess_output,
        "--system_properties", decode_param_from_env_var("system_properties"),
        "--pytorch_model_folder", pytorch_model_folder,
        "--mlflow_model_folder", mlflow_model_folder,
        "--output_model", decode_output_from_env_var('output_model')
    ]
    cmd_base.extend(cmd)
    _run_subprocess_cmd(cmd_base, component_name="finetune", completion_files_folder=completion_files_folder,
                        single_run=False, number_of_processes=num_gpus)

    # validate lora weights

    # identify model name
    model_selector_args_path = os.path.join(
        model_selector_output, SaveFileConstants.MODEL_SELECTOR_ARGS_SAVE_PATH)
    with open(model_selector_args_path, 'r') as rptr:
        model_name = json.load(rptr)['model_name']

    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.validate_lora_weights",
        "--task_name", task_name,
        "--base_pytorch_model_path", os.path.join(model_selector_output, model_name),
        "--lora_weights_path", os.path.join(pytorch_model_folder, PEFT_ADAPTER_WEIGHTS_DIR),
        "--train_file_path", os.path.join(decode_input_from_env_var("dataset_input") or "", "train_input.jsonl"),
    ]
    add_task_specific_params(cmd, task_name, component_name="validate_lora_weights")
    _run_subprocess_cmd(cmd, component_name="validate_lora_weights", completion_files_folder=completion_files_folder,
                        single_run=True, number_of_processes=num_gpus)

    # model registration
    cmd = [
        "python", "-m", "azureml.acft.contrib.hf.nlp.entry_point.finetune.register_model",
        "--task_name", task_name,
        "--model_asset_id", decode_param_from_env_var('model_asset_id'),
        "--registration_details_folder", decode_output_from_env_var('output_model'),
        "--model_path", os.path.join(
            pytorch_model_folder,
            PEFT_ADAPTER_WEIGHTS_DIR
        ),
        "--convert_to_safetensors", "true",
    ]
    add_optional_param(cmd=cmd, component_param_name="registered_model_name", argparse_param_name="model_name")
    add_optional_param(cmd=cmd, component_param_name="model_registration_tag", argparse_param_name="model_tag")
    _run_subprocess_cmd(cmd, component_name="register_model", completion_files_folder=completion_files_folder,
                        single_run=True, number_of_processes=num_gpus)


@swallow_all_exceptions(time_delay=60)
def run():
    """Run the main function."""
    # validate inputs
    FtaasPipelineInputsValidator()

    # copy the component scripts to cwd
    # _copy_components_scripts()

    # run the component script
    initiate_run()


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
