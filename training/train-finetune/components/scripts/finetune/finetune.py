# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetune component."""

import os
import json
import argparse

# from azureml.gllm.utils.metrics import get_metric
from azureml.train.finetune.core.constants import task_definitions
from azureml.train.finetune.core.drivers.finetune import run_finetune

from azureml.train.finetune.core.utils.logging_utils import get_logger_app
from azureml.train.finetune.core.constants.constants import SaveFileConstants
from azureml.train.finetune.core.utils.error_handling.exceptions import ValidationException
from azureml.train.finetune.core.utils.error_handling.error_definitions import PathNotFound
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
# from utils.decorators import swallow_all_exceptions

# Refer this logging issue
# https://github.com/Azure/azure-sdk-for-python/issues/23563

logger = get_logger_app()


def get_parser():
    """Get the parser object."""
    parser = argparse.ArgumentParser(description="Sequence classification with Lora support")

    # Model optimization settings
    parser.add_argument(
        "--apply_ort",
        type=str,
        default="false",
        help="If set to true, will use the ONNXRunTime training",
    )
    parser.add_argument(
        "--apply_deepspeed",
        type=str,
        default="false",
        help="If set to true, will enable deepspeed for training",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="./zero1.json",
        help="Deepspeed config to be used for finetuning",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        help="Local rank passed by torch distributed launch",
    )

    # Lora settings
    parser.add_argument("--apply_lora", type=str, default="false", help="lora enabled")
    parser.add_argument("--lora_alpha", type=int, default=128, help="lora attn alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="lora dropout value")
    parser.add_argument("--lora_r", default=8, type=int, help="lora dimension")
    parser.add_argument("--merge_lora_weights", type=str, default="false", help="if set to true, \
                        the lora trained weights will be merged to base model before saving")

    # Training settings
    parser.add_argument("--epochs", default=5, type=int, help="training epochs")
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
    parser.add_argument("--train_batch_size", default=4, type=int, help="Train batch size")
    parser.add_argument("--valid_batch_size", default=4, type=int, help="Validation batch size")
    parser.add_argument(
        "--auto_find_batch_size",
        default="false",
        type=str,
        help=(
            "Flag to enable auto finding of batch size. If the provided `train_batch_size` goes into Out Of Memory"
            " (OOM)enabling auto_find_batch_size will find the correct batch size by iteratively reducing"
            " `train_batch_size` by afactor of 2 till the OOM is fixed."
        ),
    )
    # -- optimizer options adamw_hf, adamw_torch, adamw_apex_fused, adafactor
    parser.add_argument(
        "--optimizer",
        default="adamw_hf",
        type=str,
        help="Optimizer to be used while training",
    )
    parser.add_argument(
        "--lr",
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
        "--evaluation_steps",
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
        type=str,
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
        "--apply_early_stopping", type=str, default="false", help="Enable early stopping"
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
        "--save_as_mlflow_model", type=str, default="true", help="Save as mlflow model with pyfunc as flavour"
    )

    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        help="output folder of preprocessor containing encoded train.jsonl, valid.jsonl, test.jsonl",
    )

    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help=("output folder of model selector containing model configs, tokenizer, checkpoints."),
    )

    parser.add_argument(
        "--output_dir_pytorch",
        default="output",
        type=str,
        help="Output dir to save the finetune model and other metadata",
    )

    parser.add_argument(
        "--output_dir_mlflow",
        default="output",
        type=str,
        help="Output dir to save the finetune model as mlflow model",
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknown_args_ = parser.parse_known_args()

    # reading the args already specified in preprocessing here
    # task_name, model_name_or_path, tokenizer settings to be used in inference
    preprocess_args_path = os.path.join(args.dataset_path, SaveFileConstants.PreprocessArgsSavePath)
    model_selector_args_path = os.path.join(args.model_path, SaveFileConstants.ModelSelectorArgsSavePath)

    if not os.path.exists(preprocess_args_path):
        raise ValidationException._with_error(AzureMLError.create(PathNotFound, path=preprocess_args_path))

    with open(preprocess_args_path, "r") as rptr:
        preprocess_args = json.load(rptr)
        args.task_name = preprocess_args.get("task_name")
        args.pad_to_max_length = preprocess_args.get("pad_to_max_length")
        args.max_seq_length = preprocess_args.get("max_seq_length")

    with open(model_selector_args_path, "r") as rptr:
        model_selector_args = json.load(rptr)
        args.model_name_or_path = model_selector_args.get("model_name_or_path")
        args.is_continual_finetuning = model_selector_args.get("continual_finetuning_model_path") is not None

    # decode the hf_task_name and hf_problem_type
    args.user_task_name = args.task_name
    task_info = getattr(task_definitions, args.user_task_name)()
    args.hf_task_name = task_info.hf_task_name
    args.hf_problem_type = task_info.hf_problem_type

    # get the num_labels
    path = os.path.join(args.dataset_path, SaveFileConstants.ClassesSavePath)
    if not os.path.exists(path):
        raise ValidationException._with_error(AzureMLError.create(PathNotFound, path=path))

    with open(path, "r") as rptr:
        class_names = json.load(rptr)[SaveFileConstants.ClassesSaveKey]
        logger.info(f"{class_names}")
        args.num_labels = len(class_names)  # used while defining the model
        args.class_names = class_names  # used while computing metrics for token classification

    # Convert the boolean variables from string to bool
    if isinstance(args.apply_lora, str):
        args.apply_lora = args.apply_lora.lower() == "true"
    if isinstance(args.apply_ort, str):
        args.apply_ort = args.apply_ort.lower() == "true"
    if isinstance(args.auto_find_batch_size, str):
        args.auto_find_batch_size = args.auto_find_batch_size.lower() == "true"
    if isinstance(args.save_as_mlflow_model, str):
        args.save_as_mlflow_model = args.save_as_mlflow_model.lower() == "true"
    if isinstance(args.merge_lora_weights, str):
        args.merge_lora_weights = args.merge_lora_weights.lower() == "true"

    # convert str to bool
    if isinstance(args.apply_deepspeed, str):
        args.apply_deepspeed = args.apply_deepspeed.lower() == "true"

    if args.apply_deepspeed and args.deepspeed_config is None:
        args.deepspeed_config = "./zero1.json"

    if args.save_as_mlflow_model:
        args.sentence1_key = preprocess_args.get("sentence1_key", None)
        args.sentence2_key = preprocess_args.get("sentence2_key", None)
        args.label_key = preprocess_args.get("label_key", None)
        args.token_key = preprocess_args.get("token_key", None)
        args.tag_key = preprocess_args.get("tag_key", None)
        args.label_all_tokens = preprocess_args.get("label_all_tokens", None)
        args.hf_task_name = preprocess_args.get("hf_task_name", None)

    args.use_fp16 = (args.precision == 16)

    if isinstance(args.resume_from_checkpoint, str):
        args.resume_from_checkpoint = args.resume_from_checkpoint.lower() == "true"

    if isinstance(args.apply_early_stopping, str):
        args.apply_early_stopping = args.apply_early_stopping.lower() == "true"

    logger.info(f"Lora Enabled: {args.apply_lora}")
    logger.info(f"ORT Enabled: {args.apply_ort}")
    logger.info(f"DeepSpeed Enabled: {args.apply_deepspeed}")
    logger.info(f"FP16 Enabled: {args.use_fp16}")
    logger.info(f"Resume from Checkpoint: {args.resume_from_checkpoint}")
    logger.info(f"Early Stopping Enabled: {args.apply_early_stopping}")
    logger.info(f"Saved as mlflow model: {args.save_as_mlflow_model}")

    if not isinstance(args.evaluation_steps_interval, float) or \
            args.evaluation_steps_interval < 0.0 or args.evaluation_steps_interval > 1.0:
        args.evaluation_steps_interval = 0.0
    else:
        logger.info(f"evaluation_steps_interval: {args.evaluation_steps_interval}")

    args.encoded_train_data_path = os.path.join(args.dataset_path, "train.jsonl")
    args.encoded_valid_data_path = os.path.join(args.dataset_path, "valid.jsonl")
    logger.info(args)

    run_finetune(args)

    # save the classes list
    classes_save_path = os.path.join(args.output_dir_pytorch, SaveFileConstants.ClassesSavePath)
    class_names_json = {SaveFileConstants.ClassesSaveKey: args.class_names}
    with open(classes_save_path, "w") as wptr:
        json.dump(class_names_json, wptr)
    logger.info(f"Classes file saved at {classes_save_path}")
