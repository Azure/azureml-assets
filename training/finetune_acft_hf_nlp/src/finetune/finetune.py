# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetune component."""

import json
import argparse
from pathlib import Path
from argparse import Namespace

from transformers.trainer_utils import set_seed, enable_full_determinism

from azureml.acft.contrib.hf.nlp.constants.constants import SaveFileConstants
from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner

from azureml.acft.accelerator.utils.run_utils import add_run_properties
from azureml.acft.accelerator.utils.decorators import swallow_all_exceptions
from azureml.acft.accelerator.utils.logging_utils import get_logger_app

from azureml.acft.accelerator.utils.error_handling.exceptions import LLMException
from azureml.acft.accelerator.utils.error_handling.error_definitions import LLMInternalError
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore

# Refer this logging issue
# https://github.com/Azure/azure-sdk-for-python/issues/23563

logger = get_logger_app()

ROOT_RUN_PROPERTIES = {
    "PipelineType": "Finetune",
}

RUN_PROPERTIES = {
    "showMetricsAtRoot": "true",
}

add_run_properties(ROOT_RUN_PROPERTIES, logger, add_to_root=True)
add_run_properties(RUN_PROPERTIES, logger)

IGNORE_MISMATCHED_SIZES_FALSE_MODELS = [
    "databricks/dolly-v2-12b",
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


def finetune(args: Namespace):
    """Finetune."""
    logger.info(f"full_determinism is set to {args.enable_full_determinism}")
    enable_full_determinism(args.seed) if args.enable_full_determinism else set_seed(args.seed)

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

    # Update the model name or path
    model_name_or_path = Path(args.model_selector_output, args.model_name)
    if model_name_or_path.is_dir():
        args.model_name_or_path = model_name_or_path
    else:
        args.model_name_or_path = args.model_name
    
    # set `ignore_mismatched_sizes` to `false` by default
    if hasattr(args, "model_name") and args.model_name in IGNORE_MISMATCHED_SIZES_FALSE_MODELS:
        logger.info(f"Forcing `ignore_mismatched_sizes` to False for {args.model_name}")
        setattr(args, "ignore_mismatched_sizes", False)

    # Below arguments are needed for HF training args
    args.output_dir = args.pytorch_model_folder
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    args.fp16 = bool(args.precision == 16)

    # Read the default deepspeed config if the apply_deepspeed is set to true without providing config file
    if args.apply_deepspeed and args.deepspeed is None:
        args.deepspeed = "./zero2.json"
    elif not args.apply_deepspeed:
        # do not use deepspeed config if provided when apply_deepspeed is set to false
        args.deepspeed = None

    if args.deepspeed:
        with open(args.deepspeed, "r") as fp:
            ds_data = json.load(fp)
        zero_optimization_config = ds_data.get("zero_optimization", {})
        ds_stage = zero_optimization_config.get("stage", None)
        # `apply_lora=true` is not supported with stage3 deepspeed config
        if ds_stage == 3 and args.apply_lora:
            raise LLMException._with_error(
                AzureMLError.create(LLMInternalError, error=(
                    "`apply_lora=true` configuration is currently not supported with deepspeed stage3 optimization"
                    )
                )
            )
        # `stage3_gather_16bit_weights_on_model_save=false` is not supported for stage3 deepspeed config
        if ds_stage == 3 and not zero_optimization_config.get("stage3_gather_16bit_weights_on_model_save", False):
            raise LLMException._with_error(
                AzureMLError.create(LLMInternalError, error=(
                    "stage3_gather_16bit_weights_on_model_save should be "
                    "`true` in deepspeed stage 3 config"
                    )
                )
            )

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

    logger.info(args)

    # Saving the args is done in `run_finetune` to handle the distributed training
    hf_task_runner = get_task_runner(task_name=args.task_name)()
    hf_task_runner.run_finetune(args)


@swallow_all_exceptions(logger)
def main():
    """Parse args and finetune."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    finetune(args)


if __name__ == "__main__":
    main()
