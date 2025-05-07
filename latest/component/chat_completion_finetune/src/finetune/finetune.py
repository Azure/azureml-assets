# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetune component."""

import os
import json
import yaml
import logging
import argparse
import shutil
from pathlib import Path
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Any
import re

import torch

# set up transformers cache
from azureml.acft.common_components.utils import transformer_utils  # noqa: F401 # Module imported but unused
from transformers.trainer_utils import set_seed, enable_full_determinism

from azureml.acft.contrib.hf.nlp.constants.constants import (
    SaveFileConstants,
    Tasks,
    HfModelTypes,
    MLFlowHFFlavourConstants,
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    MLFLOW_FLAVORS,
    SaveStrategy,
)
from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update

from azureml.acft.accelerator.utils.run_utils import add_run_properties, is_main_process
from azureml.acft.common_components.model_selector.constants import ModelSelectorDefaults
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError, ACFTSystemError
from azureml.acft.common_components.utils.mlflow_utils import update_acft_metadata
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.logging_utils import SystemSettings
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import SKUNotSupported
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.finetune.finetune")

COMPONENT_NAME = "ACFT-Finetune"

PHI3_MINI_4K_INSTRUCT_MODEL_TYPE = "phi3mini"

LLAMA_SCOUT_MODEL_TYPE = "llama4"

DEFAULT_DEEPSPEED_STAGE2_CONFIG = str(Path(__file__).parent.resolve() / "zero2.json")
DEFAULT_DEEPSPEED_STAGE3_CONFIG = str(Path(__file__).parent.resolve() / "zero3.json")


# TODO - Move REFINED_WEB to :dataclass HfModelTypes
REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models
MISTRAL = "mistral"

ROOT_RUN_PROPERTIES = {
    "PipelineType": "Finetune",
}

RUN_PROPERTIES = {
    "showMetricsAtRoot": "true",
}

add_run_properties(ROOT_RUN_PROPERTIES, add_to_root=True)
add_run_properties(RUN_PROPERTIES)


# mlflow model task based signature for inference
MLFLOW_MODEL_SIGNATURES = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: {
        "inputs": '[{"name": "input_string", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.MULTI_LABEL_CLASSIFICATION: {
        "inputs": '[{"name": "input_string", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.NAMED_ENTITY_RECOGNITION: {
        "inputs": '[{"name": "input_string", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.QUESTION_ANSWERING: {
        "inputs": '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.SUMMARIZATION: {
        "inputs": '[{"name": "input_string", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.TRANSLATION: {
        "inputs": '[{"name": "input_string", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
}


MLFLOW_MODEL_SIGNATURES_FOR_TRANSFORMERS = {
    Tasks.SINGLE_LABEL_CLASSIFICATION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
        "params": '[{"name": "return_all_scores", "dtype" : "boolean", "default" : true, "shape" : null}]',
    },
    Tasks.MULTI_LABEL_CLASSIFICATION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.NAMED_ENTITY_RECOGNITION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.QUESTION_ANSWERING: {
        "inputs": '[{"name": "question", "type": "string"}, {"name": "context", "type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.SUMMARIZATION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.TRANSLATION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
    },
    Tasks.TEXT_GENERATION: {
        "inputs": '[{"type": "string"}]',
        "outputs": '[{"type": "string"}]',
        "params": '[{"name": "top_p", "type": "float", "default": 1.0, "shape": null}, {"name": "temperature", "type": "float", "default": 0.8, "shape": null}, {"name": "max_new_tokens", "type": "integer", "default": 50, "shape": null}, {"name": "do_sample", "type": "boolean", "default": true, "shape": null}, {"name": "return_full_text", "type": "boolean", "default": true, "shape": null}]',   # noqa: E501 # Length of line greater than 119 characters limit
    },
}


MLFLOW_MODEL_SIGNATURES_FOR_FLAVOR = {
    MLFLOW_FLAVORS.TRANSFORMERS: MLFLOW_MODEL_SIGNATURES_FOR_TRANSFORMERS,
    MLFLOW_FLAVORS.HFTRANSFORMERS: MLFLOW_MODEL_SIGNATURES,
    MLFLOW_FLAVORS.HFTRANSFORMERSV2: MLFLOW_MODEL_SIGNATURES,
}


IGNORE_MISMATCHED_SIZES_FALSE_MODELS = [
    HfModelTypes.LLAMA,
    HfModelTypes.GPT_NEOX,  # dolly
    HfModelTypes.FALCON,
    HfModelTypes.REFINEDWEBMODEL,  # falcon
    HfModelTypes.MIXTRAL,
]


QLORA_SUPPORTED_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.REFINEDWEBMODEL,
    HfModelTypes.FALCON,
    REFINED_WEB,
    HfModelTypes.MIXTRAL,
    LLAMA_SCOUT_MODEL_TYPE,
]


MLFLOW_HFTRANSFORMERS_MISC_CONF = {
    # updating the parameters will override any existing misc conf keys
    HfModelTypes.LLAMA: {
        "tokenizer_hf_load_kwargs": {
            "model_input_names": ["input_ids", "attention_mask"],
        },
    },
}


ACFT_REGEX_PREFIX = "acft_regex:"

DEEPSPEED_STAGE3_SUPPORTED_TASKS = [
    Tasks.TEXT_GENERATION,
    Tasks.CHAT_COMPLETION
]
DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX_LIST = "|".join(DEEPSPEED_STAGE3_SUPPORTED_TASKS)
# the below regex exludes DEEPSPEED_STAGE3_SUPPORTED_TASKS and matches other words
DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX = f"^(?!({DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX_LIST})$)(\\w*)"


DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.FALCON,
    HfModelTypes.MISTRAL,
    HfModelTypes.MIXTRAL,
    HfModelTypes.PHI_LONGROPE,
    PHI3_MINI_4K_INSTRUCT_MODEL_TYPE,
]
DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX_LIST = "|".join(DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES)
# the below regex exludes DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES and matches other words
DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX = f"^(?!({DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX_LIST})$)(\\w*)"


FORCE_GRADIENT_CHECKPOINTING_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.FALCON,
    HfModelTypes.MISTRAL,
    HfModelTypes.MIXTRAL,
    HfModelTypes.PHI_LONGROPE,
    PHI3_MINI_4K_INSTRUCT_MODEL_TYPE,
]

FORCE_FLASH_ATTENTION_2_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.FALCON,
    HfModelTypes.MISTRAL,
    HfModelTypes.MIXTRAL,
    HfModelTypes.PHI_LONGROPE,
    PHI3_MINI_4K_INSTRUCT_MODEL_TYPE,
]


def str2bool(arg):
    """Convert string to bool."""
    arg = arg.lower()
    if arg in ["true", '1']:
        return True
    elif arg in ["false", '0']:
        return False
    else:
        raise ValueError(f"Invalid argument {arg} to while converting string to boolean")


def get_parser():
    """Get the parser object."""
    parser = argparse.ArgumentParser(description="Sequence classification with Lora support")

    # Model optimization settings
    parser.add_argument(
        "--apply_ort",
        type=str2bool,
        default="false",
        help="If set to true, will use the ONNXRunTime training",
    )
    parser.add_argument(
        "--apply_deepspeed",
        type=str2bool,
        default="false",
        help="If set to true, will enable deepspeed for training",
    )
    parser.add_argument(
        "--deepspeed_stage",
        type=int,
        default=2,
        choices=[2, 3],
        help=(
            "This parameter configures which DEFAULT deepspeed config to be used - stage2 or stage3. The default "
            "choice is stage2. Note that, this parameter is ONLY applicable when user doesn't pass any config "
            "information via deepspeed port."
        )
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Deepspeed config to be used for finetuning",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torch distributed launch",
    )

    # Lora settings
    parser.add_argument("--apply_lora", type=str2bool, default="false", help="lora enabled")
    parser.add_argument("--lora_alpha", type=int, default=128, help="lora attn alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="lora dropout value")
    parser.add_argument("--lora_r", default=8, type=int, help="lora dimension")

    # Training settings
    parser.add_argument("--num_train_epochs", default=5, type=int, help="training epochs")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=(
            "If set to a positive number, the total number of training steps to perform. Overrides `epochs`."
            "In case of using a finite iterable dataset the training may stop before reaching the set number of steps"
            "when all data is exhausted."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", default=4, type=int, help="Train batch size")
    parser.add_argument("--per_device_eval_batch_size", default=4, type=int, help="Validation batch size")
    parser.add_argument(
        "--auto_find_batch_size",
        default="false",
        type=str2bool,
        help=(
            "Flag to enable auto finding of batch size. If the provided `train_batch_size` goes into Out Of Memory"
            " (OOM) enabling auto_find_batch_size will find the correct batch size by iteratively reducing"
            " `train_batch_size` by afactor of 2 till the OOM is fixed."
        ),
    )
    # -- optimizer options adamw_torch, adamw_torch, adamw_apex_fused, adafactor
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        type=str,
        help="Optimizer to be used while training",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="Start learning rate. Defaults to linear scheduler.",
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of steps used for a linear warmup from 0 to learning_rate",
    )
    parser.add_argument(
        "--weight_decay",
        default=0,
        type=float,
        help=(
            "The weight decay to apply (if not zero) to all layers except all "
            "bias and LayerNorm weights in AdamW optimizer"
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        default=0.9,
        type=float,
        help="The beta1 hyperparameter for the AdamW optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        default=0.999,
        type=float,
        help="The beta2 hyperparameter for the AdamW optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="The epsilon hyperparameter for the AdamW optimizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1,
        type=int,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default="false",
        type=str2bool,
        help="Enable / disable gradient checkpointing",
    )
    parser.add_argument(
        "--fp16",
        default="false",
        type=str2bool,
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--bf16",
        default="false",
        type=str2bool,
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        type=str,
        help="The scheduler type to use"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        default=0,
        type=int,
        help="Number of workers to use for loading the data"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help=(
            "Apply mixed precision training. "
            "This can reduce memory footprint by performing operations in half-precision."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set at the beginning of training",
    )
    parser.add_argument(
        "--enable_full_determinism",
        type=str2bool,
        default="false",
        help="Ensure reproducible behavior during distributed training",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        type=str2bool,
        default="true",
        help=(
            "Whether or not to raise an error if some of the weights from the "
            "checkpoint do not have the same size as the weights of the model"
        ),
    )
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=3600,
        help=(
            "The timeout for `torch.distributed.init_process_group` calls, used to avoid GPU socket timeouts when "
            "performing slow operations in distributed runnings. Please refer the [PyTorch documentation] "
            "(https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more "
            "information."
        ),
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help=(
            "Maximum gradient norm (for gradient clipping)"
        ),
    )
    parser.add_argument(
        "--eval_accumulation_steps",
        default=None,
        type=int,
        help="Number of predictions steps to accumulate before moving the tensors to the CPU.",
    )
    parser.add_argument(
        "--evaluation_strategy", type=str, default="epoch", help="The evaluation strategy to adopt during training",
    )
    parser.add_argument(
        "--evaluation_steps_interval",
        type=float,
        default=0.0,
        help=(
            "The evaluation steps in fraction of an epoch steps to adopt during training. "
            "Overwrites evaluation_steps if not 0."
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of update steps between two evals if evaluation_strategy='steps'",
    )
    parser.add_argument(
        "--logging_strategy", type=str, default="epoch", help="The logging strategy to adopt during training",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of update steps between two logs if logging_strategy='steps'",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help="Specify the metric to use to compare two different models"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str2bool,
        default="false",
        help="Loads Optimizer, Scheduler and Trainer state for finetuning if true",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default=SaveStrategy.EVALUATION_STRATEGY,
        help="The checkpoint save strategy to adopt during training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Number of update steps between two checkpoint saves if save_strategy='steps'",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=-1,
        help=(
            "If a value is passed, will limit the total amount of checkpoints. "
            "Deletes the older checkpoints in output_dir. "
            "If the value is -1 saves all checkpoints"
        ),
    )

    parser.add_argument(
        "--apply_early_stopping", type=str2bool, default="false", help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Stop training when the specified metric worsens for early_stopping_patience evaluation calls",
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Denotes how much the specified metric must improve to satisfy early stopping conditions"
    )

    parser.add_argument(
        "--preprocess_output",
        default=None,
        type=str,
        help="output folder of preprocessor containing the metadata of train, evaluation and test files",
    )

    parser.add_argument(
        "--model_selector_output",
        default=None,
        type=str,
        help=("output folder of model selector containing model configs, tokenizer, checkpoints."),
    )

    parser.add_argument(
        "--pytorch_model_folder",
        default="pytorch_model_folder",
        type=str,
        help="Output dir to save the finetune model and other metadata",
    )

    parser.add_argument(
        "--mlflow_model_folder",
        default="mlflow_model_folder",
        type=str,
        help="Output dir to save the finetune model as mlflow model",
    )

    return parser


def update_lora_target_modules():
    """Update peft config with falcon target layers."""
    import peft

    models_to_lora_target_modules_map = getattr(
        peft.utils.other, "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING", {}
    )
    models_to_lora_target_modules_map.update(
        {
            HfModelTypes.REFINEDWEBMODEL: ["query_key_value"],
            REFINED_WEB: ["query_key_value"],
            HfModelTypes.FALCON: ["query_key_value"],
        }
    )
    setattr(
        peft.utils.other,
        "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
        models_to_lora_target_modules_map
    )
    setattr(
        peft.tuners.lora,
        "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
        models_to_lora_target_modules_map
    )
    logger.info(
        f"Updated lora target modules map: {peft.tuners.lora.TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING}")


def copy_preprocess_args(args: Namespace) -> Namespace:
    """Copy preprocess args to finetune."""
    # Read the preprocess component args
    # Preprocess Component + Model Selector Component ---> Finetune Component
    # Since all Model Selector Component args are saved via Preprocess Component, loading the Preprocess args
    # suffices
    preprocess_args_load_path = Path(args.preprocess_output, SaveFileConstants.PREPROCESS_ARGS_SAVE_PATH)
    with open(preprocess_args_load_path, 'r') as rptr:
        preprocess_args = json.load(rptr)
        for key, value in preprocess_args.items():
            if not hasattr(args, key):  # add keys that don't already exist
                setattr(args, key, value)

    return args


def get_deepspeed_config_json(args: Namespace) -> Dict[str, Any]:
    """Fetch deepspeed config json from file.

    :param args: User passed args
    :type Namespace
    """
    # load deepspeed config
    try:
        with open(args.deepspeed, "r", encoding="utf-8") as fp:
            ds_config_json = json.load(fp)
            return ds_config_json
    except Exception:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid deepspeed config file. Unable to load json file."
                )
            )
        )


def identify_deepspeed_stage(deepspeed_config_json: Dict[str, Any]) -> int:
    """Read the deepspeed stage from the deepspeed config."""
    zero_optimization_config = deepspeed_config_json.get("zero_optimization", {})

    ds_stage = zero_optimization_config.get("stage", None)
    if not isinstance(ds_stage, int):
        raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        "Invalid deepspeed config file. Stage information is missing from the config file."
                    )
                )
            )

    logger.info(f"Identified deepspeed stage: {ds_stage}.")

    return ds_stage


def is_match(user_value: Any, match_value: Any) -> bool:
    """Match if given user value is same value/regex match as expected match value."""
    is_match = False
    if user_value and isinstance(user_value, str) and isinstance(match_value, str) and \
            match_value.startswith(ACFT_REGEX_PREFIX):
        regex_str = match_value[len(ACFT_REGEX_PREFIX):]
        re_match = re.match(regex_str, user_value)
        if re_match is not None:
            # if there is a regex match then value is matched with value in source
            is_match = True
            logger.info(f"Regex matched: {user_value} with {regex_str}.")
        else:
            # if there is no regex match
            is_match = False
            logger.info(f"Regex not matched: {user_value} with {regex_str}.")
    else:
        is_match = bool(user_value == match_value)

    logger.info(f"Is match - {is_match}")

    return is_match


def check_for_invalid_ds_zero3_settings(args: Namespace):
    """Check if invalid ds3 settings are selected by the user.

    If fail_run is enabled for a setting raise an User Error otherwise reset the args using the valid_settings.
    :param args: User passed args
    :type Namespace
    """
    invalid_ds_zero3_settings = [
        dict(
            invalid_settings=dict(
                apply_lora=True,
                task_name=f"{ACFT_REGEX_PREFIX}{DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX}",
                model_type=f"{ACFT_REGEX_PREFIX}{DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX}",
            ),
            fail_run=True,
            valid_settings=None
        ),
        dict(
            invalid_settings=dict(apply_ort=True),
            fail_run=True,
            valid_settings=None
        ),
        dict(
            invalid_settings=dict(auto_find_batch_size=True),
            fail_run=False,
            valid_settings=dict(auto_find_batch_size=False)
        ),
        dict(  # Phi models, disable deepspeed stage 3
            invalid_settings=dict(model_type=MIXFORMER_SEQUENTIAL),
            fail_run=True,
            valid_settings=None
        )
    ]
    for setting in invalid_ds_zero3_settings:
        invalid_settings = setting["invalid_settings"]
        fail_run = setting["fail_run"]
        valid_settings = setting["valid_settings"]
        if all([is_match(getattr(args, key, None), value) for key, value in invalid_settings.items()]):
            if fail_run:
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Invalid settings found. Deep Speed stage3 doesn't work with {invalid_settings}."
                        )
                    )
                )
            else:
                if valid_settings is None:
                    raise ACFTValidationException._with_error(
                        AzureMLError.create(
                            ACFTSystemError,
                            pii_safe_message="Valid settings cannot be None."
                        )
                    )
                logger.info(
                    "Found invalid settings with deepspeed stage3."
                    f"Reconfiguring the user parameters: {valid_settings}."
                )
                for key, value in valid_settings.items():
                    setattr(args, key, value)


def _set_hf_trainer_args_from_finetune_config(args: Namespace, finetune_config: Dict[str, Any]):
    """Read :param `hf_trainer_args` from finetune config and set them to args."""
    hf_trainer_args = finetune_config.get("hf_trainer_args", {})
    for arg_name, arg_value in hf_trainer_args.items():
        setattr(args, arg_name, arg_value)
        logger.info(f"Setting {arg_name} to {arg_value}")


def validate_ds_zero3_config(deepspeed_config_json: Dict[str, Any]):
    """Validate the deepspeed zero3 config file.

    :param deepspeed_config: path to the deepspeed config file
    :type str
    """
    zero_optimization_config = deepspeed_config_json.get("zero_optimization", {})

    if not zero_optimization_config.get("stage3_gather_16bit_weights_on_model_save", False):
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "stage3_gather_16bit_weights_on_model_save should be "
                    "`true` in deepspeed stage 3 config."
                )
            )
        )


def setup_deepspeed_nebula(ds_config_json: Dict[str, Any], pytorch_model_folder: str,
                           model_name_or_path: str) -> Dict[str, Any]:
    """Set nebula settings in ds config if it has been enabled."""
    nebula: Dict = ds_config_json.get("nebula", {})
    if not nebula:
        return ds_config_json
    enabled = nebula.get("enabled", False)
    if not enabled:
        del ds_config_json["nebula"]
        return ds_config_json
    nebula_dirname = "nebula_checkpoints"
    nebula["persistent_storage_path"] = os.path.abspath(os.path.join(pytorch_model_folder, nebula_dirname))
    nebula["persistent_time_interval"] = nebula.get("persistent_time_interval", 30)
    nebula["num_of_version_in_retention"] = nebula.get("num_of_version_in_retention", 2)
    nebula["enable_nebula_load"] = True
    logger.info(f"Nebula settings: {nebula}")

    model_name_or_path = Path(model_name_or_path)
    if model_name_or_path.is_dir():
        logger.info(f"Copying checkpoints from {model_name_or_path} to {pytorch_model_folder}...")
        try:
            shutil.copytree(model_name_or_path, pytorch_model_folder, dirs_exist_ok=True)
        except Exception as e:
            shutil.rmtree(pytorch_model_folder, ignore_errors=True)
            raise ACFTValidationException._with_error(
                AzureMLError.create(ACFTSystemError, pii_safe_message=f"shutil copy failed with err: {e}"))

    ds_config_json["nebula"] = nebula
    return ds_config_json


def is_vllm_enabled(task_name: str, finetune_config: Dict[str, Any]) -> bool:
    """Read :flag `inferencing_config.enable_vllm` to enable vllm for finetuned model.

    1. vllm support is disabled by default.
    2. To enable vllm support update the param :inferencing_config.enable_vllm to True.
    3. Legacy support
        vllm support for text generation task is enabled by default.
        To disable it set the param :inferencing_config.enable_vllm to False.
    """
    if (
        finetune_config.get("inferencing_config") is not None and
        finetune_config["inferencing_config"].get("enable_vllm") is not None
    ):
        enable_vllm = bool(finetune_config["inferencing_config"]["enable_vllm"])
        enabled_or_disabled_warn_msg = "enabled" if enable_vllm else "disabled"
        logger.warning(f"Vllm inferencing is {enabled_or_disabled_warn_msg} for {task_name} from finetune config.")
        return enable_vllm
    # legacy support for already supported models
    elif task_name == Tasks.TEXT_GENERATION:
        logger.warning(
            f"Vllm inferencing is auto enabled for {task_name}. "
            "Set the :param `inferencing_config.enable_vllm to False` to disable it."
        )
        return True
    return False  # default case


def setup_vllm(task_name: str, finetune_config: Dict[str, Any], base_model_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Enable/disable vllm for finetuned model inferencing."""
    if not is_vllm_enabled(task_name, finetune_config):
        removed_base_image = base_model_metadata.pop("azureml.base_image", None)
        if removed_base_image is not None:
            logger.warning(f"Removed base image meta data for mitigation of FT model not deployable issue, \
                        base image value is {removed_base_image}.")
    else:
        if base_model_metadata.get("azureml.base_image") is not None:
            # FT environment for chat-completion task is updated to transformers 4.46.3 and
            # FTed model tokenizers can't be loaded with transformers<4.45.0
            # Hence FMI version < 61 can't be used for deployments and Inferencing.
            # Currently force updating the image in finetune script.
            # After models are migrated to mcr.microsoft.com/azureml/curated/foundation-model-inference:61
            # remove the following code marked for removal, including comments
            # start of removal code
            if task_name == Tasks.CHAT_COMPLETION:
                try:
                    base_vllm_image = str(base_model_metadata.get("azureml.base_image"))
                    base_vllm_container, base_vllm_image_version = base_vllm_image.split(":")
                    if int(base_vllm_image_version) < 61:
                        logger.info("Updating the vllm inference container version.")
                        base_model_metadata["azureml.base_image"] = base_vllm_container + ":61"
                except Exception:
                    logger.info("Unable to fetch vllm inference container version, force updating the image.")
                    base_model_metadata["azureml.base_image"] = \
                        "mcr.microsoft.com/azureml/curated/foundation-model-inference:61"
            # end of removal code
            logger.info(
                "Adding inferencing base image {} for {} task.\
                ".format(base_model_metadata.get("azureml.base_image"), task_name)
            )
    return base_model_metadata


def resolve_deepspeed_config(args: Namespace) -> str:
    """Identify the right deepspeed config to be used based on user passed parameters."""
    # Check for deepspeed config via input port
    if getattr(args, "deepspeed", None) is not None:
        logger.info(f"Found deepspeed config via input port - {args.deepspeed}.")
        return args.deepspeed

    default_deepspeed_config = (
        DEFAULT_DEEPSPEED_STAGE2_CONFIG
        if args.deepspeed_stage == 2 else
        DEFAULT_DEEPSPEED_STAGE3_CONFIG
    )
    logger.info(f"Using default deepspeed config: {default_deepspeed_config}")
    return default_deepspeed_config


def setup_and_validate_deepspeed(args: Namespace, do_validate: bool = True):
    """Deepspeed initialization and validation.

    :param args: User passed args
    :type Namespace
    :param do_validate: Validates the deepspeed config file in case of deepspeed stage3
    :type bool
    """
    logger.info("Setting up deespeed.")
    # Read the default deepspeed config if the apply_deepspeed is set to true without providing config file
    args.deepspeed = resolve_deepspeed_config(args)
    if args.deepspeed is None:
        logger.info("Deepspeed is not enabled. Nothing to setup!")
        return

    # Validate auto_find_batch_size
    if args.auto_find_batch_size:
        raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            "Invalid settings found. Deep Speed cannot be coupled with auto_find_batch_size.\n"
                            "1. If you want to use auto_find_batch_size functionality set apply_deepspeed to false\n"
                            "2. Otherwise, set auto_find_batch_size to false and use per_device_train_batch_size of 1"
                        )
                    )
                )
    # load deepspeed config
    ds_config_json = get_deepspeed_config_json(args)

    ds_stage = identify_deepspeed_stage(ds_config_json)
    # set proper deespeeed stage in finetune args
    # so that down stream components like model_converter can use proper values if to merge model or not
    setattr(args, "deepspeed_stage", ds_stage)

    # add validations for deepspeed stage3
    if do_validate and ds_stage == 3:
        # activate few deepspeed stage3 specific configurations
        enable_ds3_model_specific_args(args)
        # validate the ds config file
        logger.info("Validating deepspeed config.")
        validate_ds_zero3_config(ds_config_json)
        # check for invalid settings
        logger.info("Checking for invalid deepspeed configurations.")
        check_for_invalid_ds_zero3_settings(args)

    ds_config_json = setup_deepspeed_nebula(ds_config_json, args.pytorch_model_folder, args.model_name_or_path)
    args.deepspeed = ds_config_json  # replace file path with updated dict


def enable_ds3_model_specific_args(args: Namespace):
    """Override or enable few model specific parameters.

    Invoke the function only when deepspeed stage3 is enabled.
    """
    pass


def set_16bit_precision(args: Namespace):
    """Set fp16/bf16 in args based on cuda device support."""
    if torch.cuda.is_bf16_supported():
        args.bf16 = True
        logger.info("Setting bfloat16 to True.")
    else:
        args.fp16 = True
        logger.info("Setting float16 to True.")


def set_flash_attention(args: Namespace):
    """Set Flash Attention related parameters."""
    flash_attention_load_model_kwargs = {}
    if (
        hasattr(args, "model_type")
        and args.model_type in FORCE_FLASH_ATTENTION_2_MODEL_TYPES
    ):
        # only Ampere or higher architecture supports Flash attention 2
        # Flash attention 2 is supported with 16-bit, 8-bit anf 4-bit
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and args.precision in [16, 8, 4]:
            # `use_flash_attention_2=True` will be deprecated, use `attn_implementation="flash_attention_2"`
            flash_attention_load_model_kwargs.update({"attn_implementation": "flash_attention_2"})
            setattr(args, "apply_flash_attention", True)
            setattr(args, "flash_attention_version", 2)
        # elif args.precision == 16:
        #     # Flash attention is supported with only 16-bit
        #     setattr(args, "apply_flash_attention", True)
        #     setattr(args, "flash_attention_version", 1)
        # else:
        #     # unable to use Flash attention as precision is not supported
        #     logger.warning(f"{args.precision}-bit precision is not supported for Flash attention.")
        #     logger.warning("Disabling Flash attention.")
        #     setattr(args, "apply_flash_attention", False)
        #     setattr(args, "flash_attention_version", -1)
        else:
            logger.warning("Flash Attention is not supported on current compute.")
            setattr(args, "apply_flash_attention", False)
            setattr(args, "flash_attention_version", -1)
        if args.flash_attention_version != -1:
            # Set 16-bit precision value in Quantization case for Flash Attention to work.
            # Currently will fail with error `RuntimeError: FlashAttention only support fp16 and bf16 data type`.
            # When fp16/bf16 is set the attention q,k,v layers are autocasted to respective precision from `uint8`.
            if (args.finetune_in_4bit or args.finetune_in_8bit) and not (args.fp16 or args.bf16):
                set_16bit_precision(args)
            # Flash attention is supported only when model is loaded in respective supported precision
            if args.bf16:
                flash_attention_load_model_kwargs.update({"torch_dtype": torch.bfloat16})
            elif args.fp16:
                flash_attention_load_model_kwargs.update({"torch_dtype": torch.float16})
            # update finetune_config to load model with flash_attention_2/torch_dtype
            args.finetune_config = deep_update(
                args.finetune_config,
                {
                    "load_model_kwargs": flash_attention_load_model_kwargs,
                }
            )
    else:
        setattr(args, "apply_flash_attention", False)
        setattr(args, "flash_attention_version", -1)
    if args.precision == 32 and (args
                                 .finetune_config
                                 .get("load_model_kwargs", {})
                                 .get("use_flash_attention_2", False) is True):
        # Flash attention is not supported with 32-bit precision
        logger.warning("Flash Attention is not supported with 32-bit precision.")
        raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            "Flash Attention is not supported with 32-bit precision."
                        )
                    )
                )

    logger.info(f"enable Flash attention: {getattr(args, 'apply_flash_attention', None)}")
    logger.info(f"Using Flash Attention version: {getattr(args, 'flash_attention_version', None)}")
    logger.info(f"Flash Attention model load kwargs: {flash_attention_load_model_kwargs}")


def set_gradient_checkpointing(args: Namespace):
    """Set Gradient checkpointing related parameters."""
    if args.apply_lora and not args.apply_deepspeed:
        # do not set `gradient_checkpointing` for LoRA only training as it fails with the following error:
        # RuntimeError: Expected to mark a variable ready only once. This error is caused by one of the following
        # reasons: 1) Use of a module parameter outside the `forward` function. Please make sure model parameters
        # are not shared across multiple concurrent forward-backward passes. or try to use _set_static_graph() as
        # a workaround if this module graph does not change during training loop.2) Reused parameters in multiple
        # reentrant backward passes. For example, if you use multiple `checkpoint` functions to wrap the same part
        # of your model, it would result in the same set of parameters been used by different reentrant backward
        # passes multiple times, and hence marking a variable ready multiple times. DDP does not support such use
        # cases in default. You can try to use _set_static_graph() as a workaround if your module graph does not
        # change over iterations.
        # Parameter at index xxx has been marked as ready twice. This means that multiple autograd engine  hooks
        # have fired for this particular parameter during this iteration. You can set the environment variable
        # TORCH_DISTRIBUTED_DEBUG to either INFO or DETAIL to print parameter names for further debugging.
        logger.info("Not setting `gradient_checkpointing` to True for LoRA only finetuning.")
        return

    if (
        hasattr(args, "model_type")
        and args.model_type in FORCE_GRADIENT_CHECKPOINTING_MODEL_TYPES
    ):
        logger.info(
            f"Identified model type: {args.model_type}. Forcing `gradient_checkpointing` to True."
        )
        setattr(args, "gradient_checkpointing", True)

    logger.info(f"enable Gradient checkpointing: {getattr(args, 'gradient_checkpointing', None)}")


def setup_automl_nlp(args: Namespace) -> None:
    """Set automl nlp related args."""
    if args.task_name in [Tasks.NLP_MULTICLASS, Tasks.NLP_MULTILABEL, Tasks.NLP_NER]:
        # Disable adding prefixes to logger for NLP Tasks.
        args.set_log_prefix = False
        logger.info(f"Using log prefix - {args.set_log_prefix}")


def _load_mlflow_model(model_path: str) -> str:
    mlflow_config_file = Path(model_path, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    mlmodel_data = None
    if mlflow_config_file.is_file():
        try:
            with open(mlflow_config_file, "r") as rptr:
                mlmodel_data = yaml.safe_load(rptr)
        except Exception as e:
            logger.info(f"Unable to load MLmodel file - {e}")
    else:
        logger.info("MLmodel file does not exist")
    return mlmodel_data


def _get_model_flavor(mlflow_flavors: list, mlmodel_data: dict) -> str:
    for each_flavor in mlflow_flavors:
        if each_flavor in mlmodel_data["flavors"]:
            logger.info(f"Current mlflow flavor - {each_flavor}")
            return each_flavor
    logger.info("MLmodel file does not have any mlflow flavor")
    return None


def validate_learning_rate(args: Namespace) -> None:
    """Validate learning rate."""
    if args.learning_rate <= 0:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid learning rate. Learning rate should be greater than 0."
                )
            )
        )


def finetune(args: Namespace):
    """Finetune."""
    logger.info(f"full_determinism is set to {args.enable_full_determinism}")
    enable_full_determinism(args.seed) if args.enable_full_determinism else set_seed(args.seed)

    # Update the model name or path
    model_name_or_path = Path(args.model_selector_output, args.model_name)
    if model_name_or_path.is_dir():
        args.model_name_or_path = model_name_or_path
    else:
        args.model_name_or_path = args.model_name

    # fetch model asset id
    model_asset_id = getattr(args, "model_asset_id", None) or ""

    # additional logging
    logger.info(f"Model name: {getattr(args, 'model_name', None)}")
    logger.info(f"Task name: {getattr(args, 'task_name', None)}")
    logger.info(f"Model asset id: {model_asset_id}")
    logger.info(f"enable LoRA: {getattr(args, 'apply_lora', None)}")
    logger.info(f"enable DeepSpeed: {getattr(args, 'apply_deepspeed', None)}")
    logger.info(f"enable ORT: {getattr(args, 'apply_ort', None)}")
    logger.info(f"Precision: {getattr(args, 'precision', None)}")

    # set `ignore_mismatched_sizes` to `false` by default
    if (
        hasattr(args, "model_type")
        and args.model_type in IGNORE_MISMATCHED_SIZES_FALSE_MODELS
    ):
        logger.info(
            f"Identified model type: {args.model_type}. Forcing `ignore_mismatched_sizes` to False."
        )
        setattr(args, "ignore_mismatched_sizes", False)

    # set eval_accumulation_steps to None if passed a non-positive value
    eval_accumulation_steps = getattr(args, "eval_accumulation_steps", -1)
    if eval_accumulation_steps and eval_accumulation_steps <= 0:
        setattr(args, "eval_accumulation_steps", None)

    logger.info(f"eval_accumulation_steps: {getattr(args, 'eval_accumulation_steps', None)}")

    # read FT config
    ft_config_path = Path(args.model_selector_output, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)
    if ft_config_path.is_file():
        with open(ft_config_path, "r") as rptr:
            ft_config = json.load(rptr)
            setattr(args, "finetune_config", ft_config)
            logger.info("Added finetune config to `component_args`")
            # Read the lora parameters from finetune config
            if "lora_algo" in ft_config:
                logger.info(f'Setting lora_algo to: {ft_config.get("lora_algo")}')
                setattr(args, "lora_algo", ft_config.get("lora_algo"))
            if "lora_target_modules" in ft_config:
                logger.info(f'Setting lora_target_modules to: {ft_config.get("lora_target_modules")}')
                setattr(args, "lora_target_modules", ft_config.get("lora_target_modules"))
            # Read leaf modules for MoE models from finetune config
            if "leaf_modules_of_moe_models" in ft_config:
                logger.info(f'Setting leaf_modules_of_moe_models to: {ft_config.get("leaf_modules_of_moe_models")}')
                setattr(args, "leaf_modules_of_moe_models", ft_config.get("leaf_modules_of_moe_models"))
            # Reading hf trainer args from finetune config
            _set_hf_trainer_args_from_finetune_config(args, ft_config)
    else:
        logger.info(f"{SaveFileConstants.ACFT_CONFIG_SAVE_PATH} does not exist")
        setattr(args, "finetune_config", {})

    # `mlflow_ft_conf` - contains all mlflow related properties
    mlflow_ft_conf = {
        "mlflow_model_signature": {},
        "mlflow_hftransformers_misc_conf": {},
        "mlflow_flavor": None,
    }

    mlmodel_data = _load_mlflow_model(args.model_selector_output)
    mlflow_flavor = None
    if mlmodel_data is not None:
        mlflow_flavors = [
            MLFLOW_FLAVORS.TRANSFORMERS,
            MLFLOW_FLAVORS.HFTRANSFORMERS,
            MLFLOW_FLAVORS.HFTRANSFORMERSV2,
        ]
        mlflow_flavor = _get_model_flavor(mlflow_flavors, mlmodel_data)
        mlflow_ft_conf["mlflow_flavor"] = mlflow_flavor
        # set task based mlflow_model_signature
        if getattr(args, "task_name", None) is not None:
            if mlflow_flavor is not None and mlflow_flavor in MLFLOW_MODEL_SIGNATURES_FOR_FLAVOR.keys():
                if args.task_name in MLFLOW_MODEL_SIGNATURES_FOR_FLAVOR[mlflow_flavor]:
                    mlflow_ft_conf["mlflow_model_signature"] = deep_update(
                        mlflow_ft_conf["mlflow_model_signature"],
                        MLFLOW_MODEL_SIGNATURES_FOR_FLAVOR[mlflow_flavor][args.task_name],
                    )
                    logger.info(
                                f"Adding mlflow model signature for task {args.task_name} - "
                                f"{MLFLOW_MODEL_SIGNATURES_FOR_FLAVOR[mlflow_flavor][args.task_name]}"
                            )

    # set `mlflow_flavor` in finetune args
    setattr(args, "mlflow_flavor", mlflow_flavor)

    # remove mlflow_model_signature if empty
    if "mlflow_model_signature" in mlflow_ft_conf \
            and len(mlflow_ft_conf["mlflow_model_signature"]) == 0:
        del mlflow_ft_conf["mlflow_model_signature"]

    model_name_or_type = None
    # pass `mlflow_hftransformers_misc_conf` to be set in mlflow model
    if hasattr(args, "model_name") and args.model_name in MLFLOW_HFTRANSFORMERS_MISC_CONF:
        model_name_or_type = args.model_name
    if hasattr(args, "model_type") and args.model_type in MLFLOW_HFTRANSFORMERS_MISC_CONF:
        model_name_or_type = args.model_type
    if model_name_or_type is not None:
        mlflow_hftransformers_misc_conf = MLFLOW_HFTRANSFORMERS_MISC_CONF[model_name_or_type]
        logger.info(
            f"Forcing `mlflow_hftransformers_misc_conf` to set to {mlflow_hftransformers_misc_conf} "
            f"for {model_name_or_type}"
        )
        mlflow_ft_conf["mlflow_hftransformers_misc_conf"] = deep_update(
            mlflow_ft_conf["mlflow_hftransformers_misc_conf"],
            mlflow_hftransformers_misc_conf,
        )

    metadata = {}
    # if MLmodel file exists pass to finetuned model as `base_model_mlmodel`
    mlflow_config_file = Path(args.model_selector_output, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    if mlflow_config_file.is_file():
        import yaml
        mlflow_data = None
        try:
            with open(mlflow_config_file, "r") as rptr:
                mlflow_data = yaml.safe_load(rptr)
                metadata = mlflow_data.get("metadata", {})
        except Exception as e:
            logger.info(f"Unable to load MLmodel file - {e}")
        if mlflow_data is not None:
            # pass base model MLmodel file data if available
            mlflow_hftransformers_misc_conf = mlflow_ft_conf.get("mlflow_hftransformers_misc_conf", {})
            mlflow_hftransformers_misc_conf.update({"base_model_mlmodel": mlflow_data})
            mlflow_ft_conf["mlflow_hftransformers_misc_conf"] = deep_update(
                mlflow_ft_conf["mlflow_hftransformers_misc_conf"],
                mlflow_hftransformers_misc_conf,
            )
            logger.info(f"Setting `base_model_mlmodel` in finetuned mlflow model - {mlflow_hftransformers_misc_conf}")
        else:
            logger.info("MLmodel file is empty")
    else:
        logger.info("MLmodel file does not exist")
    if mlmodel_data is not None:
        # pass base model MLmodel file data if available
        mlflow_hftransformers_misc_conf = mlflow_ft_conf.get("mlflow_hftransformers_misc_conf", {})
        mlflow_hftransformers_misc_conf.update({"base_model_mlmodel": mlmodel_data})
        mlflow_ft_conf["mlflow_hftransformers_misc_conf"] = deep_update(
            mlflow_ft_conf["mlflow_hftransformers_misc_conf"],
            mlflow_hftransformers_misc_conf,
        )
        logger.info(f"Setting `base_model_mlmodel` in finetuned mlflow model - {mlflow_hftransformers_misc_conf}")
    else:
        logger.info("MLmodel file is empty")

    # if input is pytorch model, read metadata if the metadata.json exists.
    if not metadata:
        metadatapath = os.path.join(model_name_or_path, ModelSelectorDefaults.MODEL_DEFAULTS_PATH)
        if os.path.isfile(metadatapath):
            with open(metadatapath, "r") as rptr:
                metadata = json.load(rptr)

    logger.info(f"FT MLFlow config - {mlflow_ft_conf}")

    mlflow_ft_conf = deep_update(mlflow_ft_conf, args.finetune_config.get("mlflow_ft_conf", {}))
    args.finetune_config["mlflow_ft_conf"] = deepcopy(mlflow_ft_conf)
    logger.info(f"Updated FT MLFlow config - {args.finetune_config['mlflow_ft_conf']}")

    # Below arguments are needed for HF training args
    args.output_dir = args.pytorch_model_folder
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    if args.precision == 16:
        set_16bit_precision(args)
    args.finetune_in_8bit = bool(args.precision == 8)  # 8 bit finetune
    args.finetune_in_4bit = bool(args.precision == 4)  # 4 bit finetune

    # set flash-attention
    set_flash_attention(args)

    # set gradient-checkpointing
    set_gradient_checkpointing(args)

    validate_learning_rate(args)

    if args.finetune_in_8bit or args.finetune_in_4bit:
        if hasattr(args, "model_type") and args.model_type not in QLORA_SUPPORTED_MODEL_TYPES:
            raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(
                            f"Quantized finetune is not supported for model family: {args.model_type}."
                        )
                    )
                )
        logger.info("Enabling QLoRA finetuning")
        if not args.apply_lora:
            logger.info("Lora is not enabled. Setting it to true.")
            setattr(args, "apply_lora", True)
        if args.apply_deepspeed:
            logger.info(
                "Deepspeed is enabled which is not compatible with QLoRA. "
                "Resetting Deepspeed to false."
            )
            setattr(args, "apply_deepspeed", False)
        if args.gradient_checkpointing:
            logger.info(
                "Gradient checkpointing is enabled which is not compatible with QLoRA. "
                "Resetting Gradient checkpointing to false."
            )
            setattr(args, "gradient_checkpointing", False)

    setattr(args, "apply_ort", can_apply_ort(args, logger))

    # Deepspeed enabled
    if args.apply_deepspeed:
        setup_and_validate_deepspeed(args)
    else:
        # do not use deepspeed config if provided when apply_deepspeed is set to false
        args.deepspeed = None

    if (
        not isinstance(args.evaluation_steps_interval, float) or
        args.evaluation_steps_interval < 0.0 or
        args.evaluation_steps_interval > 1.0
    ):
        args.evaluation_steps_interval = 0.0
    else:
        logger.info(f"evaluation_steps_interval: {args.evaluation_steps_interval}")

    if args.save_strategy == SaveStrategy.EVALUATION_STRATEGY:
        logger.info(f"Setting save strategy to evaluation strategy: {args.evaluation_strategy}, {args.eval_steps}")
        args.save_strategy = args.evaluation_strategy
        args.save_steps = args.eval_steps

    # setup vllm for finetuned model inference
    metadata = setup_vllm(args.task_name, args.finetune_config, metadata)

    args.model_metadata = update_acft_metadata(metadata=metadata,
                                               finetuning_task=args.task_name,
                                               base_model_asset_id=model_asset_id)

    setup_automl_nlp(args)

    # Saving the args is done in `run_finetune` to handle the distributed training
    hf_task_runner = get_task_runner(task_name=args.task_name)()
    hf_task_runner.run_finetune(args)

    # post-training execute any code on main-process only to avoid race conditions.
    if is_main_process():
        # copy conda file
        conda_file_path = Path(args.model_selector_output, MLFlowHFFlavourConstants.CONDA_YAML_FILE)
        if conda_file_path.is_file():
            shutil.copy(str(conda_file_path), args.output_dir)
            logger.info(f"Copied {MLFlowHFFlavourConstants.CONDA_YAML_FILE} file to output dir.")

        # copy pre-processor config files
        preprocessor_config_file = Path(args.model_selector_output, "default_model_name", "preprocessor_config.json")
        if preprocessor_config_file.is_file():
            shutil.copy(str(preprocessor_config_file), args.output_dir)
            logger.info("Copied preprocessor_config.json file to output dir.")
        # copy inference config files
        mlflow_ml_configs_dir = Path(args.model_selector_output, "ml_configs")
        ml_config_dir = Path(args.output_dir, "ml_configs")
        if mlflow_ml_configs_dir.is_dir():
            shutil.copytree(
                mlflow_ml_configs_dir,
                ml_config_dir
            )
            logger.info("Copied ml_configs folder to output dir.")


def can_apply_ort(args: Namespace, logger):
    """Can ORT be enabled."""
    if args.apply_ort and args.task_name in (Tasks.SUMMARIZATION, Tasks.TRANSLATION):
        logger.warning("Enabling ORT has a breaking change with summarization and translation tasks "
                       "so diabling ORT for SUMMARIZATION and TRANSLATION tasks")
        return False
    logger.warning("Disabling ORT for all tasks")
    return False


@swallow_all_exceptions(time_delay=60)
def main():
    """Parse args and finetune."""
    if not torch.cuda.is_available():
        raise ACFTValidationException._with_error(AzureMLError.create(SKUNotSupported))

    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Copy the args generated in the preprocess step
    args = copy_preprocess_args(args)

    # Appending the global rank to log file generated by each process. This is to avoid issues in multi-node runs
    # where having the same file name in each node is causing issues during file upload.
    SystemSettings.LOG_FILENAME = SystemSettings.LOG_FILENAME + f'.{os.environ["RANK"]}'
    # Set logging parameters
    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
            LoggingLiterals.COMPONENT_NAME: COMPONENT_NAME,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
        log_level=logging.INFO,
    )

    # XXX Hack to support model loading in accelerator package for falcon models
    # with `trust_remote_code=True`
    # This is needed as FT config is ONLY used in contrib package which is causing
    # failure when loading the model in accelerator as part of peft weights merge
    if hasattr(args, "model_type") and args.model_type in [
        HfModelTypes.REFINEDWEBMODEL,
        HfModelTypes.FALCON,
        REFINED_WEB,
        MIXFORMER_SEQUENTIAL
    ]:
        from functools import partial
        from transformers.models.auto import (
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForCausalLM,
        )

        # Updata lora target modules for falcon
        update_lora_target_modules()

        AutoModelForSequenceClassification.from_pretrained = partial(
            AutoModelForSequenceClassification.from_pretrained, trust_remote_code=True
        )
        AutoModelForTokenClassification.from_pretrained = partial(
            AutoModelForTokenClassification.from_pretrained, trust_remote_code=True
        )
        AutoModelForQuestionAnswering.from_pretrained = partial(
            AutoModelForQuestionAnswering.from_pretrained, trust_remote_code=True
        )
        AutoModelForCausalLM.from_pretrained = partial(
            AutoModelForCausalLM.from_pretrained, trust_remote_code=True
        )
        logger.info("Updated `from_pretrained` method for Seq cls, Tok cls, QnA and Text Gen")

    # XXX Hack to support loading best peft model after finetuning to ignore base model layers.
    # This is needed for transformers==4.37.2 and is already fixed in transformers>=4.38.0.
    # Currently transformers==4.38.1 has issues with multi-node training, hence not upgrading transformers further.
    if getattr(args, "apply_lora", False) and getattr(args, "apply_deepspeed", False):
        from functools import partialmethod
        from deepspeed import DeepSpeedEngine

        DeepSpeedEngine.load_checkpoint = partialmethod(DeepSpeedEngine.load_checkpoint, load_module_strict=False)
        logger.info("Updated `DeepSpeedEngine.load_checkpoint` defaults to use `load_module_strict=False`.")

    finetune(args)


if __name__ == "__main__":
    main()
