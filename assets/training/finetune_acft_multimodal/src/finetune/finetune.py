# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
File containing function for finetune component.
"""

import json
import argparse
from pathlib import Path
from argparse import Namespace

from transformers.trainer_utils import (
    set_seed, 
    enable_full_determinism,
    IntervalStrategy
)

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.model_selector.constants import ModelSelectorConstants
from azureml.acft.common_components.utils.constants import MlflowMetaConstants
from azureml.acft.common_components.utils.mlflow_utils import update_acft_metadata
from azureml.acft.common_components.utils.arg_utils import str2bool
from azureml.acft.multimodal.components import PROJECT_NAME, VERSION
from azureml.acft.multimodal.components.constants.constants import SaveFileConstants, Tasks, \
    ModelTypes, MMEFTHyperParameterDefaults, ProblemType
from azureml.acft.multimodal.components.task_factory import get_task_runner

logger = get_logger_app("azureml.acft.multimodal.components.scripts.components.finetune_mmeft.finetune")


def get_parser():
    """
    Get the parser object.
    """

    parser = argparse.ArgumentParser(description="Multimodal single label classification with Lora support")

    # Model optimization settings
    parser.add_argument(
        "--apply_ort",
        type=lambda x: bool(str2bool(str(x), "apply_ort")),
        default=False,
        help="If set to true, will use the ONNXRunTime training"
    )
    parser.add_argument(
        "--apply_deepspeed",
        type=lambda x: bool(str2bool(str(x), "apply_deepspeed")),
        default=False,
        help="If set to true, will enable deepspeed for training"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Deepspeed config to be used for finetuning"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed by torch distributed launch"
    )

    # Lora settings
    parser.add_argument(
        "--apply_lora",
        type=lambda x: bool(str2bool(str(x), "apply_lora")),
        default=False,
        help="lora enabled"
    )
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
        )
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        default=MMEFTHyperParameterDefaults.TRAINING_BATCH_SIZE,
        type=int,
        help="Train batch size"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        default=MMEFTHyperParameterDefaults.VALIDATION_BATCH_SIZE,
        type=int,
        help="Validation batch size"
    )
    parser.add_argument(
        "--auto_find_batch_size",
        type=lambda x: bool(str2bool(str(x), "auto_find_batch_size")),
        default=False,
        help=(
            "Flag to enable auto finding of batch size. If the provided `train_batch_size` goes into Out Of Memory"
            " (OOM)enabling auto_find_batch_size will find the correct batch size by iteratively reducing"
            " `train_batch_size` by afactor of 2 till the OOM is fixed."
        )
    )
    parser.add_argument(
        "--problem_type",
        default=ProblemType.SINGLE_LABEL_CLASSIFICATION,
        type=str,
        help="Whether its single label or multilabel classification"
    )
    # -- optimizer options adamw_hf, adamw_torch, adamw_apex_fused, adafactor
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        type=str,
        help="Optimizer to be used while training"
    )
    parser.add_argument(
        "--learning_rate",
        default=MMEFTHyperParameterDefaults.LEARNING_RATE,
        type=float,
        help="Start learning rate. Defaults to linear scheduler."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Number of steps used for a linear warmup from 0 to learning_rate"
    )
    parser.add_argument(
        "--warmup_ratio",
        default=MMEFTHyperParameterDefaults.WARMUP_RATIO,
        type=float,
        help="Number of steps used for a linear warmup from 0 to learning_rate"
    )
    parser.add_argument(
        "--weight_decay",
        default=0,
        type=float,
        help=(
            "The weight decay to apply (if not zero) to all layers except all "
            "bias and LayerNorm weights in AdamW optimizer"
        )
    )
    parser.add_argument(
        "--adam_beta1",
        default=0.9,
        type=float,
        help="The beta1 hyperparameter for the AdamW optimizer"
    )
    parser.add_argument(
        "--adam_beta2",
        default=0.999,
        type=float,
        help="The beta2 hyperparameter for the AdamW optimizer"
    )
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="The epsilon hyperparameter for the AdamW optimizer"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=MMEFTHyperParameterDefaults.GRADIENT_ACCUMULATION_STEPS,
        type=int,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass"
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
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set at the beginning of training"
    )
    parser.add_argument(
        "--enable_full_determinism",
        type=lambda x: bool(str2bool(str(x), "enable_full_determinism")),
        default=False,
        help="Ensure reproducible behavior during distributed training"
    )
    parser.add_argument(
        "--ignore_mismatched_sizes",
        type=lambda x: bool(str2bool(str(x), "ignore_mismatched_sizes")),
        default=False,
        help=(
            "Whether or not to raise an error if some of the weights from the "
            "checkpoint do not have the same size as the weights of the model"
        )
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm (for gradient clipping)"
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="epoch",
        choices=(
            IntervalStrategy.NO,
            IntervalStrategy.STEPS,
            IntervalStrategy.EPOCH,
        ),
        help="The evaluation strategy to adopt during training"
    )
    parser.add_argument(
        "--evaluation_steps_interval",
        type=float,
        default=0.0,
        help=(
            "The evaluation steps in fraction of an epoch steps to adopt during training. "
            "Overwrites evaluation_steps if not 0."
        )
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of update steps between two evals if evaluation_strategy='steps'"
    )
    parser.add_argument(
        "--logging_strategy", type=str, default="epoch", help="The logging strategy to adopt during training"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of update steps between two logs if logging_strategy='steps'"
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help="Specify the metric to use to compare two different models"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=lambda x: bool(str2bool(str(x), "resume_from_checkpoint")),
        default=False,
        help="Loads Optimizer, Scheduler and Trainer state for finetuning if true"
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=-1,
        help=(
            "If a value is passed, will limit the total amount of checkpoints. "
            "Deletes the older checkpoints in output_dir. "
            "If the value is -1 saves all checkpoints"
        )
    )

    parser.add_argument(
        "--apply_early_stopping",
        type=lambda x: bool(str2bool(str(x), "apply_early_stopping")),
        default=False,
        help="Enable early stopping"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Stop training when the specified metric worsens for early_stopping_patience evaluation calls"
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Denotes how much the specified metric must improve to satisfy early stopping conditions"
    )

    parser.add_argument(
        "--save_as_mlflow_model",
        type=lambda x: bool(str2bool(str(x), "save_as_mlflow_model")),
        default=True,
        help="Save as mlflow model with pyfunc as flavour"
    )

    parser.add_argument(
        "--preprocess_output",
        default=None,
        type=str,
        help="output folder of preprocessor containing the metadata of train, evaluation and test files"
    )

    parser.add_argument(
        "--model_selector_output",
        default=None,
        type=str,
        help="output folder of model selector containing model configs, tokenizer, checkpoints."
    )

    parser.add_argument(
        "--pytorch_model_folder",
        default=SaveFileConstants.DEFAULT_PYTORCH_OUTPUT,
        type=str,
        help="Output dir to save the finetune model and other metadata"
    )

    parser.add_argument(
        "--mlflow_model_folder",
        default=SaveFileConstants.DEFAULT_MLFLOW_OUTPUT,
        type=str,
        help="Output dir to save the finetune model as mlflow model"
    )

    return parser


def finetune(args: Namespace):
    """
    Main function handling finetune.
    """

    logger.info(f"full_determinism is set to {args.enable_full_determinism}")
    enable_full_determinism(args.seed) if args.enable_full_determinism else set_seed(args.seed)

    # Read the preprocess component args
    # Preprocess Component + Model Selector Component ---> Finetune Component
    # Since all Model Selector Component args are saved via Preprocess Component, loading the Preprocess args
    # suffices
    preprocess_args_load_path = Path(args.preprocess_output, SaveFileConstants.PREPROCESS_ARGS_SAVE_PATH)
    with open(preprocess_args_load_path, "r") as rptr:
        preprocess_args = json.load(rptr)
        for key, value in preprocess_args.items():
            if not hasattr(args, key):  # add keys that don"t already exist
                setattr(args, key, value)

    # Update the model name or path
    model_name_or_path = Path(args.model_selector_output, args.model_name)
    if model_name_or_path.is_dir():
        args.model_name_or_path = model_name_or_path
    else:
        args.model_name_or_path = args.model_name

    # Below arguments are needed for HF training args
    args.output_dir = SaveFileConstants.DEFAULT_OUTPUT_DIR
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    args.fp16 = bool(args.precision == 16)

    # Read the default deepspeed config if the apply_deepspeed is set to true without providing config file
    if args.apply_deepspeed and args.deepspeed is None:
        args.deepspeed = "./zero1.json"
    elif not args.apply_deepspeed:
        # do not use deepspeed config if provided when apply_deepspeed is set to false
        args.deepspeed = None

    if not isinstance(args.evaluation_steps_interval, float) or \
            args.evaluation_steps_interval < 0.0 or args.evaluation_steps_interval > 1.0:
        args.evaluation_steps_interval = 0.0
    else:
        logger.info(f"evaluation_steps_interval: {args.evaluation_steps_interval}")

    args.save_strategy = args.evaluation_strategy
    args.save_steps = args.eval_steps
    args.load_best_model_at_end = True
    logger.info(args)

    # Saving the args is done in `run_finetune` to handle the distributed training
    args.model_type = ModelTypes.MMEFT

    # can remove this once common model selector code is used for model import
    if not hasattr(args, ModelSelectorConstants.MODEL_METADATA):
        args.model_metadata = {MlflowMetaConstants.BASE_MODEL_NAME :args.model_name}
    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_finetune_for_mmeft(args)


if __name__ == "__main__":
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type=Tasks.MUTIMODAL_CLASSIFICATION,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        }
    )

    finetune(args)
