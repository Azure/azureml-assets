# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetune component."""

import os
import json
import logging
import argparse
from pathlib import Path
from argparse import Namespace
from copy import deepcopy
from typing import Dict, Any
import re

import torch

from transformers.trainer_utils import set_seed, enable_full_determinism

from azureml.acft.contrib.hf.nlp.constants.constants import (
    SaveFileConstants,
    Tasks,
    HfModelTypes,
    MLFlowHFFlavourConstants,
    LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    MLFLOW_FLAVORS,
)
from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner
from azureml.acft.contrib.hf.nlp.utils.common_utils import deep_update

from azureml.acft.accelerator.utils.run_utils import add_run_properties
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import ACFTUserError, ACFTSystemError
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.logging_utils import SystemSettings
from azureml.acft.contrib.hf import VERSION, PROJECT_NAME

from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import (
    swallow_all_exceptions,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import SKUNotSupported
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

logger = get_logger_app("azureml.acft.contrib.hf.scripts.components.scripts.finetune.finetune")

COMPONENT_NAME = "ACFT-Finetune"

DEFAULT_DEEPSPEED_STAGE2_CONFIG = Path(__file__).parent.resolve() / "zero2.json"
DEFAULT_DEEPSPEED_STAGE3_CONFIG = Path(__file__).parent.resolve() / "zero3.json"


# TODO - Move REFINED_WEB to :dataclass HfModelTypes
REFINED_WEB = "RefinedWeb"
MIXFORMER_SEQUENTIAL = "mixformer-sequential"  # Phi models

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
]


QLORA_SUPPORTED_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.REFINEDWEBMODEL,
    HfModelTypes.FALCON,
    REFINED_WEB,
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
]
DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX_LIST = "|".join(DEEPSPEED_STAGE3_SUPPORTED_TASKS)
# the below regex exludes DEEPSPEED_STAGE3_SUPPORTED_TASKS and matches other words
DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX = f"^(?!({DEEPSPEED_STAGE3_SUPPORTED_TASKS_REGEX_LIST})$)(\\w*)"


DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.FALCON,
]
DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX_LIST = "|".join(DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES)
# the below regex exludes DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES and matches other words
DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX = f"^(?!({DEEPSPEED_STAGE3_SUPPORTED_MODEL_TYPES_REGEX_LIST})$)(\\w*)"


FORCE_GRADIENT_CHECKPOINTING_MODEL_TYPES = [
    HfModelTypes.LLAMA,
    HfModelTypes.FALCON,
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
    # -- optimizer options adamw_hf, adamw_torch, adamw_apex_fused, adafactor
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
        default="output",
        type=str,
        help="Output dir to save the finetune model and other metadata",
    )

    parser.add_argument(
        "--mlflow_model_folder",
        default="output",
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

    # load deepspeed config
    ds_config_json = get_deepspeed_config_json(args)

    # add validations for deepspeed stage3
    if do_validate and identify_deepspeed_stage(ds_config_json) == 3:
        # activate few deepspeed stage3 specific configurations
        enable_ds3_model_specific_args(args)
        # validate the ds config file
        logger.info("Validating deepspeed config.")
        validate_ds_zero3_config(ds_config_json)
        # check for invalid settings
        logger.info("Checking for invalid deepspeed configurations.")
        check_for_invalid_ds_zero3_settings(args)


def enable_ds3_model_specific_args(args: Namespace):
    """Override or enable few model specific parameters.

    Invoke the function only when deepspeed stage3 is enabled.
    """
    if (
        hasattr(args, "model_type")
        and args.model_type in FORCE_GRADIENT_CHECKPOINTING_MODEL_TYPES
    ):
        logger.info(
            f"Identified model type: {args.model_type}. Forcing `gradient_checkpointing` to True."
        )
        setattr(args, "gradient_checkpointing", True)


def setup_automl_nlp(args: Namespace) -> None:
    """Set automl nlp related args."""
    if args.task_name in [Tasks.NLP_MULTICLASS, Tasks.NLP_MULTILABEL, Tasks.NLP_NER]:
        # Disable adding prefixes to logger for NLP Tasks.
        args.set_log_prefix = False
        logger.info(f"Using log prefix - {args.set_log_prefix}")


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
    if getattr(args, "eval_accumulation_steps", -1) <= 0:
        setattr(args, "eval_accumulation_steps", None)

    logger.info(f"eval_accumulation_steps: {getattr(args, 'eval_accumulation_steps', None)}")

    # read FT config
    ft_config_path = Path(args.model_selector_output, SaveFileConstants.ACFT_CONFIG_SAVE_PATH)
    if ft_config_path.is_file():
        with open(ft_config_path, "r") as rptr:
            ft_config = json.load(rptr)
            setattr(args, "finetune_config", ft_config)
            logger.info("Added finetune config to `component_args`")
    else:
        logger.info(f"{SaveFileConstants.ACFT_CONFIG_SAVE_PATH} does not exist")
        setattr(args, "finetune_config", {})

    # `mlflow_ft_conf` - contains all mlflow related properties
    mlflow_ft_conf = {
        "mlflow_model_signature": {},
        "mlflow_hftransformers_misc_conf": {},
    }

    # set task based mlflow_model_signature
    if getattr(args, "task_name", None) is not None and args.task_name in MLFLOW_MODEL_SIGNATURES:
        mlflow_ft_conf["mlflow_model_signature"] = deep_update(
            mlflow_ft_conf["mlflow_model_signature"],
            MLFLOW_MODEL_SIGNATURES[args.task_name],
        )
        logger.info(
                    f"Adding mlflow model signature for task {args.task_name} - "
                    f"{MLFLOW_MODEL_SIGNATURES[args.task_name]}"
                )

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

    # if MLmodel file exists pass to finetuned model as `base_model_mlmodel`
    mlflow_config_file = Path(args.model_selector_output, MLFlowHFFlavourConstants.MISC_CONFIG_FILE)
    if mlflow_config_file.is_file():
        import yaml
        mlflow_data = None
        try:
            with open(mlflow_config_file, "r") as rptr:
                mlflow_data = yaml.safe_load(rptr)
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

    logger.info(f"FT MLFlow config - {mlflow_ft_conf}")

    mlflow_ft_conf = deep_update(mlflow_ft_conf, args.finetune_config.get("mlflow_ft_conf", {}))
    args.finetune_config["mlflow_ft_conf"] = deepcopy(mlflow_ft_conf)
    logger.info(f"Updated FT MLFlow config - {args.finetune_config['mlflow_ft_conf']}")

    # Below arguments are needed for HF training args
    args.output_dir = args.pytorch_model_folder
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    if args.precision == 16:
        if torch.cuda.is_bf16_supported():
            args.bf16 = True
            logger.info("Setting bfloat16 to True.")
        else:
            args.fp16 = True
            logger.info("Setting float16 to True.")
    args.finetune_in_8bit = bool(args.precision == 8)  # 8 bit finetune
    args.finetune_in_4bit = bool(args.precision == 4)  # 4 bit finetune

    if args.finetune_in_8bit or args.finetune_in_4bit:
        if hasattr(args, "model_type") and args.model_type not in QLORA_SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Looks like quantized finetune is enabled for model family: {args.model_type} which is not supported")
        logger.info("Enabling QLoRA finetuning")
        if not args.apply_lora:
            logger.info("Lora is not enabled. Setting it to true.")
            setattr(args, "apply_lora", True)
        if args.apply_deepspeed:
            logger.info(
                "Deepspeed is enabled which is not compatible with QLoRA. "
                "Resetting Deepspeed to false"
            )
            setattr(args, "apply_deepspeed", False)

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

    args.save_strategy = args.evaluation_strategy
    args.save_steps = args.eval_steps

    setup_automl_nlp(args)

    # Saving the args is done in `run_finetune` to handle the distributed training
    hf_task_runner = get_task_runner(task_name=args.task_name)()
    hf_task_runner.run_finetune(args)


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

    finetune(args)


if __name__ == "__main__":
    main()
