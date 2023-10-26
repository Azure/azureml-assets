# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for finetune component."""

import os
import json
import argparse
from transformers.training_args import OptimizerNames
from transformers.trainer_utils import (
    SchedulerType,
    IntervalStrategy
)
from optimum.onnxruntime.training_args import ORTOptimizerNames
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.acft.accelerator.constants import MetricConstants

from azureml.acft.common_components.model_selector.constants import ModelSelectorDefaults
from azureml.acft.common_components.utils.error_handling.exceptions import ACFTValidationException
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ModelInputEmpty, ArgumentInvalid, ACFTUserError
)
from azureml.acft.common_components.utils.arg_utils import str2bool
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import swallow_all_exceptions

from azureml.metrics import constants as metrics_constants

from azureml.acft.image import VERSION, PROJECT_NAME
from azureml.acft.image.components.common.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS
from azureml.acft.image.components.finetune.common.constants.constants import (
    SettingParameters
)
from azureml.acft.image.components.finetune.factory.mappings import MODEL_FAMILY_CLS
from azureml.acft.image.components.finetune.factory.task_definitions import Tasks
from azureml.acft.image.components.finetune.finetune_runner import finetune_runner
from azureml.acft.image.components.finetune.defaults.training_defaults import (
    TrainingDefaults,
)
from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals

logger = get_logger_app("azureml.acft.image.scripts.components.finetune.finetune")


class IncomingLearingScheduler:
    """Learning scheduler names incoming from the UI. These names are such that we align with current runtime."""

    WARMUP_LINEAR = "warmup_linear"
    WARMUP_COSINE = "warmup_cosine"
    WARMUP_COSINE_WITH_RESTARTS = "warmup_cosine_with_restarts"
    WARMUP_POLYNOMIAL = "warmup_polynomial"
    CONSTANT = "constant"
    WARMUP_CONSTANT = "warmup_constant"
    STEP = "step"


class IncomingOptimizerNames:
    """Optimizer names incoming from the UI. These names are such that we align with current runtime."""

    ADAMW_TORCH = "adamw"
    ADAMW_HF = "adamw_hf"
    ADAMW_TORCH_XLA = "adamw_torch_xla"
    ADAMW_APEX_FUSED = "adamw_apex_fused"
    ADAMW_BNB = "adamw_bnb_8bit"
    ADAMW_ANYPRECISION = "adamw_anyprecision"
    SGD = "sgd"
    ADAFACTOR = "adafactor"
    ADAGRAD = "adagrad"
    ADAMW_ORT_FUSED = "adamw_ort_fused"


class Mapper:
    """Mapper class to map incoming names to the names used in the finetune runner."""

    # Learning rate scheduler mapping from incoming names to finetune runner names
    LR_SCHEDULER_MAP = {
        IncomingLearingScheduler.WARMUP_LINEAR: SchedulerType.LINEAR,
        IncomingLearingScheduler.WARMUP_COSINE: SchedulerType.COSINE,
        IncomingLearingScheduler.WARMUP_COSINE_WITH_RESTARTS: SchedulerType.COSINE_WITH_RESTARTS,
        IncomingLearingScheduler.WARMUP_POLYNOMIAL: SchedulerType.POLYNOMIAL,
        IncomingLearingScheduler.CONSTANT: SchedulerType.CONSTANT,
        IncomingLearingScheduler.WARMUP_CONSTANT: SchedulerType.CONSTANT_WITH_WARMUP
    }

    # Optimizer names mapping from incoming names to finetune runner names
    OPTIMIZER_MAP = {
        IncomingOptimizerNames.ADAMW_TORCH: OptimizerNames.ADAMW_TORCH,
        IncomingOptimizerNames.ADAMW_HF: OptimizerNames.ADAMW_HF,
        IncomingOptimizerNames.ADAMW_TORCH_XLA: OptimizerNames.ADAMW_TORCH_XLA,
        IncomingOptimizerNames.ADAMW_APEX_FUSED: OptimizerNames.ADAMW_APEX_FUSED,
        IncomingOptimizerNames.ADAMW_BNB: OptimizerNames.ADAMW_BNB,
        IncomingOptimizerNames.ADAMW_ANYPRECISION: OptimizerNames.ADAMW_ANYPRECISION,
        IncomingOptimizerNames.SGD: OptimizerNames.SGD,
        IncomingOptimizerNames.ADAFACTOR: OptimizerNames.ADAFACTOR,
        IncomingOptimizerNames.ADAGRAD: OptimizerNames.ADAGRAD,
        IncomingOptimizerNames.ADAMW_ORT_FUSED: ORTOptimizerNames.ADAMW_ORT_FUSED
    }


def get_parser():
    """Get the parser object."""
    parser = argparse.ArgumentParser(description="Image Tasks which include Image classification, \
                                                  object detection and instance segmentation.")

    # # component input: model path from model_selector component
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help=(
            "output folder of model selector containing model configs, checkpoints in case"
            "model_path is provided as input to model selector component."
            "If model_name is provided as input to model selector component,"
            "the model download happens dynamically on the fly."
        )
    )

    # # Component input: Training and validation dataset
    parser.add_argument(
        "--train_mltable_path",
        type=str,
        required=True,
        help="Path to the mltable of the training dataset."
    )
    parser.add_argument(
        "--valid_mltable_path",
        type=str,
        default=None,
        help="Path to the mltable of the validation dataset."
    )

    # # Image height and width
    parser.add_argument(
        "--image_width",
        type=int,
        default=-1,
        help="Final Image width after augmentation that is input to the network. \
            Default value is -1 which means it would be overwritten by default image \
            width in Hugging Face feature extractor. If either image_width or image_height \
            is set to -1, default value would be used for both width and height."
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=-1,
        help="Final Image height after augmentation that is input to the network. \
            Default value is -1 which means it would be overwritten by default image \
            height in Hugging Face feature extractor. If either image_width or image_height \
            is set to -1, default value would be used for both width and height."
    )

    # Image min_size and max_size. Only applicable for OD and IS.
    parser.add_argument(
        "--image_min_size",
        type=int,
        help="Minimum image size after augmentation that is input to the network. Default \
            is -1 which means it would be overwritten by image_scale in model config. \
            The image will be rescaled as large as possible within \
            the range [image_min_size, image_max_size]. \
            The image size will be constraint so that the max edge is no longer than \
            image_max_size and short edge is no longer than image_min_size."
    )
    parser.add_argument(
        "--image_max_size",
        type=int,
        help="Maximum image size after augmentation that is input to the network. Default \
            is -1 which means it would be overwritten by image_scale in model config. \
            The image will be rescaled as large as possible within\
            the range [image_min_size, image_max_size]. \
            The image size will be constraint so that the max edge is no longer than \
            image_max_size and short edge is no longer than image_min_size."
    )

    # # Task name
    parser.add_argument(
        "--task_name",
        type=str,
        choices=(
            Tasks.HF_MULTI_CLASS_IMAGE_CLASSIFICATION,
            Tasks.HF_MULTI_LABEL_IMAGE_CLASSIFICATION,
            Tasks.MM_OBJECT_DETECTION,
            Tasks.MM_INSTANCE_SEGMENTATION,
            Tasks.MM_MULTI_OBJECT_TRACKING
        ),
        required=True,
        help="Name of the task the model is solving."
    )

    # # Apply augmentaions
    parser.add_argument(
        "--apply_augmentations",
        type=lambda x: bool(str2bool(str(x), "apply_augmentations")),
        default=False,
        help="If set to true, will enable augmentations for training"
    )

    # # Data loader num workers
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help=(
            "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be "
            "loaded in the main process."
        )
    )

    # # Deepspeed
    parser.add_argument(
        "--apply_deepspeed",
        type=lambda x: bool(str2bool(str(x), "apply_deepspeed")),
        help="If set to true, will enable deepspeed for training. "
        "If left empty, will be chosen automatically based on the task type and model selected."
    )
    # optional component input: deepspeed config json
    # core is using this parameter to check if deepspeed is enabled
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="Deepspeed config to be used for finetuning",
    )

    # # ORT
    parser.add_argument(
        "--apply_ort",
        type=lambda x: bool(str2bool(str(x), "apply_ort")),
        help="If set to true, will enable Onnxruntime for training. "
        "If left empty, will be chosen automatically based on the task type and model selected."
    )

    # # LORA
    # Lora is not supported for vision models currently.
    # So, this parameter is not exposed via yaml
    parser.add_argument(
        "--apply_lora",
        type=lambda x: bool(str2bool(str(x), "apply_lora")),
        default=False,
        help="If set to true, will enable LORA for training."
    )
    parser.add_argument("--lora_alpha", type=int, default=128, help="LORA attention alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LORA dropout value")
    parser.add_argument("--lora_r", default=8, type=int, help="LORA dimension")

    # # Epochs and steps
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        help=(
            "Number of training epochs."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        help=(
            "If set to a positive number, the total number of training steps to perform. Overrides `num_train_epochs`."
            "In case of using a finite iterable dataset the training may stop before reaching the set number of steps"
            "when all data is exhausted."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )

    # # Batch size
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Train batch size. If left empty, will be chosen automatically based on the task type and model selected."
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        help=(
            "Validation batch size."
            "If left empty, will be chosen automatically based on the task type and model selected."
        )
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

    # # optimizer
    parser.add_argument(
        "--optim",
        choices=(
            IncomingOptimizerNames.ADAMW_HF,
            IncomingOptimizerNames.ADAMW_TORCH,
            # # # Todo: enable or take them out post testing.
            # IncomingOptimizerNames.ADAMW_TORCH_XLA,
            # IncomingOptimizerNames.ADAMW_APEX_FUSED,
            # IncomingOptimizerNames.ADAMW_BNB,
            # IncomingOptimizerNames.ADAMW_ANYPRECISION,
            IncomingOptimizerNames.SGD,
            IncomingOptimizerNames.ADAFACTOR,
            IncomingOptimizerNames.ADAGRAD,
            IncomingOptimizerNames.ADAMW_ORT_FUSED,
        ),
        type=str,
        help=(
            "Optimizer to be used while training."
            f"'{IncomingOptimizerNames.ADAMW_ORT_FUSED}' is only supported for ORT training."
            "If left empty, will be chosen automatically based on the task type and model selected."
        )
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help=(
            "The weight decay to apply (if not zero) to all layers except all "
            "bias and LayerNorm weights in AdamW optimizer."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )
    parser.add_argument(
        "--extra_optim_args",
        default="",
        type=str,
        help=("Optional additional arguments that are supplied to SGD Optimizer."
              "The arguments should be semi-colon separated key value pairs. "
              "For example, 'momentum=0.5; nesterov=True' for sgd"
              "Please make sure to use a valid parameter names for the chosen optimizer. For exact parameter names"
              "please refer https://pytorch.org/docs/1.13/generated/torch.optim.SGD.html#torch.optim.SGD for SGD."
              "Parameters supplied in extra_optim_args will take precedence over the parameter supplied via"
              "other arguments such as weight_decay. If weight_decay is provided via 'weight_decay'"
              "parameter and via extra_optim_args both, values specified in extra_optim_args will be used."
              ),
    )
    # # Learning rate
    parser.add_argument(
        "--learning_rate",
        type=float,
        help=(
            "Start learning rate. Defaults to 5e-05."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )

    # # Learning rate scheduler
    parser.add_argument(
        "--lr_scheduler_type",
        choices=(
            IncomingLearingScheduler.WARMUP_LINEAR,
            IncomingLearingScheduler.WARMUP_COSINE,
            IncomingLearingScheduler.WARMUP_COSINE_WITH_RESTARTS,
            IncomingLearingScheduler.WARMUP_POLYNOMIAL,
            IncomingLearingScheduler.CONSTANT,
            IncomingLearingScheduler.WARMUP_CONSTANT,
            IncomingLearingScheduler.STEP,
            # Step is not supported for FT-components. We are only accepting step as a parameter to show
            # a better warning message to the user.
            # "Step" LR scheduler is supported for runtime component therefore if user is using "Step" scheduler
            # for the uber component, the FT-component might get this value and
            # we will warn the user to use a different scheduler.
        ),
        type=str,
        help=(
            "The scheduler type to use."
            "If left empty, will be chosen automatically based on the task type and model selected."
        )
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        help=(
            "Number of steps used for a linear warmup from 0 to learning_rate."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )

    # # Gradient accumulation steps
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help=(
            "Number of updates steps to accumulate the gradients for, before performing a backward/update pass."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )

    # # mixed precision training
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        choices=(16, 32),
        help=(
            "Apply mixed precision training. "
            "This can reduce memory footprint by performing operations in half-precision."
        )
    )

    # # random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that will be set at the beginning of training."
    )

    # # evaluation
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        choices=(
            IntervalStrategy.NO,
            IntervalStrategy.STEPS,
            IntervalStrategy.EPOCH,
        ),
        default="epoch",
        help="The evaluation strategy to adopt during training."
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Number of update steps between two evals if evaluation_strategy='steps'."
    )

    # # logging
    parser.add_argument(
        "--logging_strategy",
        type=str,
        choices=(
            IntervalStrategy.NO,
            IntervalStrategy.STEPS,
            IntervalStrategy.EPOCH,
        ),
        default="epoch",
        help="The logging strategy to adopt during training."
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Number of update steps between two logs if logging_strategy='steps'."
    )

    # # save strategy
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=(
            IntervalStrategy.NO,
            IntervalStrategy.STEPS,
            IntervalStrategy.EPOCH,
        ),
        default="epoch",
        help="The checkpoint save strategy to adopt during training."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of updates steps before two checkpoint saves if save_strategy='steps'."
    )

    # # checkpoint saving limit
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=-1,
        help=(
            "If a value is passed, will limit the total amount of checkpoints. "
            "Deletes the older checkpoints in output_dir. "
            "If the value is -1 saves all checkpoints."
        ),
    )

    # # metrics for model
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        choices=(
            "loss",
            # Classification - multiclass
            metrics_constants.F1_MACRO,
            metrics_constants.ACCURACY,
            metrics_constants.PRECISION_MACRO,
            metrics_constants.RECALL_MACRO,
            # Classification - multilabel
            metrics_constants.IOU,
            metrics_constants.IOU_MACRO,
            metrics_constants.IOU_MICRO,
            metrics_constants.IOU_WEIGHTED,
            # Object detectaion and instance segmentation
            metrics_constants.MEAN_AVERAGE_PRECISION,
            metrics_constants.PRECISION,
            metrics_constants.RECALL,
            metrics_constants.MOTA,
            metrics_constants.MOTP,
            metrics_constants.IDF1,
            metrics_constants.IDSW
        ),
        help=(
            "Specify the metric to use to compare two different models."
            "If left empty, will be chosen automatically based on the task type and model selected."
        ),
    )
    # label smoothing factor
    parser.add_argument(
        "--label_smoothing_factor",
        type=float,
        help=(
            "The label smoothing factor to use in range [0.0, 1,0). Zero means no label smoothing, "
            "otherwise the underlying onehot-encoded labels are changed from 0s and 1s to "
            "label_smoothing_factor/num_labels and "
            "1 - label_smoothing_factor + label_smoothing_factor/num_labels respectively."
            "If left empty, will be chosen automatically based on the task type and model selected."
        )
    )

    # # to resume training from a model given in folder, loading older states etc.
    parser.add_argument(
        "--resume_from_checkpoint",
        type=lambda x: bool(str2bool(str(x), "resume_from_checkpoint")),
        default=False,
        help="Loads Optimizer, Scheduler and Trainer state for finetuning if true."
    )

    # # early stopping - enabled through a callback ?
    parser.add_argument(
        "--apply_early_stopping",
        type=lambda x: bool(str2bool(str(x), "apply_early_stopping")),
        default=False,
        help="Enable early stopping."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=1,
        help="Stop training when the specified metric worsens for early_stopping_patience evaluation calls."
    )
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.0,
        help="Denotes how much the specified metric must improve to satisfy early stopping conditions."
    )

    # Gradient norm
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        help=(
            "Maximum gradient norm (for gradient clipping)"
            "If left empty, will be chosen automatically based on the task type and model selected."
        )
    )

    # # Model saving - will always be set to True for vision models
    parser.add_argument(
        "--save_as_mlflow_model",
        type=lambda x: bool(str2bool(str(x), "save_as_mlflow_model")),
        default=True,
        help="Save as mlflow model with pyfunc as flavour."
    )

    # # component output: output dir for mlflow model
    parser.add_argument(
        "--mlflow_model_folder",
        default=SettingParameters.DEFAULT_MLFLOW_OUTPUT,
        type=str,
        help="Output dir to save the finetune model as mlflow model."
    )
    parser.add_argument(
        "--pytorch_model_folder",
        default=SettingParameters.DEFAULT_PYTORCH_OUTPUT,
        type=str,
        help="Output dir to save the finetune model as pytorch model."
    )

    # ############### MMDetection specific args #################### #
    parser.add_argument(
        "--iou_threshold",
        type=float,
        help="IOU threshold used during inference in non-maximum suppression post processing."
    )
    parser.add_argument(
        "--box_score_threshold",
        type=float,
        help=(
            "During inference, only return proposals with a score greater than `box_score_threshold`. "
            "The score is the multiplication of the objectness score and classification probability."
        )
    )

    return parser


@swallow_all_exceptions(time_delay=60)
def main():
    """Driver function."""
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # step learning rate scheduler can only come from sweep component. The only other option that's available in
    # sweep components is warmup_cosine, hence we are raising following exception.
    if args.lr_scheduler_type == IncomingLearingScheduler.STEP:
        error_string = f"Step scheduler is not supported by Huggingface and MMdetection trainer. Please choose "\
            f"{IncomingLearingScheduler.WARMUP_COSINE} as the learning rate scheduler."
        raise ACFTValidationException._with_error(
            AzureMLError.create(ACFTUserError, pii_safe_message=error_string)
        )

    set_logging_parameters(
        task_type=args.task_name,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    # Read the preprocess component args
    # Preprocess Component + Model Selector Component ---> Finetune Component
    # Since all Model Selector Component args are saved via Preprocess Component, loading the Preprocess args
    # suffices
    model_selector_args_save_path = os.path.join(args.model_path, ModelSelectorDefaults.MODEL_SELECTOR_ARGS_SAVE_PATH)
    with open(model_selector_args_save_path, "r") as rptr:
        preprocess_args = json.load(rptr)
        for key, value in preprocess_args.items():
            if not hasattr(args, key):  # update the values that don't already exist
                logger.info(f"{key}, {value}")
                setattr(args, key, value)

    # continual_finetuning is when an existing model which is already registered in the workspace
    # is used for finetuning. In that case, model_name would be None and either of the
    # pytorch_model_path or mlflow_model_path would be present.
    # In case of continual finetuning, We need to set model_name_or_path to the pyroch/mlflow model path.
    if hasattr(args, "pytorch_model_path") and args.pytorch_model_path:
        args.model_name_or_path = os.path.join(args.model_path, args.pytorch_model_path)
        args.is_continual_finetuning = True
        if args.model_family in (MODEL_FAMILY_CLS.MMDETECTION_IMAGE, MODEL_FAMILY_CLS.MMTRACKING_VIDEO):
            args.model_weights_path_or_url = os.path.join(args.model_path, args.model_weights_path_or_url)
            args.model_metafile_path = os.path.join(args.model_path, args.model_metafile_path)
    elif hasattr(args, "mlflow_model_path") and args.mlflow_model_path:
        args.model_name_or_path = os.path.join(args.model_path, args.mlflow_model_path)
        args.is_continual_finetuning = True
        if args.model_family in (MODEL_FAMILY_CLS.MMDETECTION_IMAGE, MODEL_FAMILY_CLS.MMTRACKING_VIDEO):
            args.model_metafile_path = os.path.join(args.model_path, args.model_metafile_path)
            args.model_weights_path_or_url = os.path.join(args.model_path, args.model_weights_path_or_url)
    elif hasattr(args, "model_name") and args.model_name:
        args.model_name_or_path = args.model_name
        args.is_continual_finetuning = False
        if args.model_family in (MODEL_FAMILY_CLS.MMDETECTION_IMAGE, MODEL_FAMILY_CLS.MMTRACKING_VIDEO):
            args.model_weights_path_or_url = args.model_weights_path_or_url
    else:
        raise ACFTValidationException._with_error(
              AzureMLError.create(ModelInputEmpty,
                                  argument_name="Model ports and model_name"
                                  )
              )

    # Map learing rate scheduler to as expected by the Trainer class
    if args.lr_scheduler_type is not None:
        args.lr_scheduler_type = Mapper.LR_SCHEDULER_MAP[args.lr_scheduler_type]

    # Map optimizer to as expected by the Trainer class
    if args.optim is not None:
        args.optim = Mapper.OPTIMIZER_MAP[args.optim]

    # Update 'args' namespace with defaults based on task type and model selected
    # Doing this before any assignment to 'args' namespace
    if args.task_name in [
        Tasks.MM_OBJECT_DETECTION,
        Tasks.MM_INSTANCE_SEGMENTATION,
        Tasks.HF_MULTI_CLASS_IMAGE_CLASSIFICATION,
        Tasks.HF_MULTI_LABEL_IMAGE_CLASSIFICATION,
        Tasks.MM_MULTI_OBJECT_TRACKING,
    ]:
        training_defaults = TrainingDefaults(
            task=args.task_name,
            model_name_or_path=args.model_name_or_path,
        )
        # Update the namespace object with values from the dictionary
        # Only update the values that don't already exist or are None
        for key, value in training_defaults.defaults_dict.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)

    logger.info(f"Using learning rate scheduler - {args.lr_scheduler_type}")
    logger.info(f"Using optimizer - {args.optim}")

    if args.task_name == Tasks.HF_MULTI_LABEL_IMAGE_CLASSIFICATION and \
            args.label_smoothing_factor is not None and args.label_smoothing_factor > 0.0:
        args.label_smoothing_factor = 0.0
        msg = f"Label smoothing is not supported for multi-label image classification. \
            Setting label_smoothing_factor to 0.0 from {args.label_smoothing_factor}"
        logger.warning(msg)

    if args.task_name in [
        Tasks.MM_OBJECT_DETECTION,
        Tasks.MM_INSTANCE_SEGMENTATION,
        Tasks.MM_MULTI_OBJECT_TRACKING
    ]:
        # Note: This is temporary check to disable deepspeed and ORT for MM tasks, until they are working.
        deepspeed_ort_arg_names = []
        if args.apply_deepspeed is True:
            deepspeed_ort_arg_names.append("apply_deepspeed")
        if args.apply_ort is True:
            deepspeed_ort_arg_names.append("apply_ort")
        if len(deepspeed_ort_arg_names) >= 1:
            deepspeed_ort_arg_names = ",".join(deepspeed_ort_arg_names)
            err_msg = f"{deepspeed_ort_arg_names} not yet supported for {args.task_name}, will be enabled in future."
            raise ACFTValidationException._with_error(
                AzureMLError.create(ArgumentInvalid, argument_name=f"{deepspeed_ort_arg_names}", expected_type=err_msg)
            )

    if args.apply_ort is False and args.optim == IncomingOptimizerNames.ADAMW_ORT_FUSED:
        error_string = f"ORT fused AdamW ({IncomingOptimizerNames.ADAMW_ORT_FUSED}) optimizer \
        should only be used with ORT training."
        raise ACFTValidationException._with_error(
            AzureMLError.create(ACFTUserError, pii_safe_message=error_string)
        )

    if args.apply_ort is True and args.optim != IncomingOptimizerNames.ADAMW_ORT_FUSED:
        logger.warning(f"ORT training is enabled but optimizer is not set to {IncomingOptimizerNames.ADAMW_ORT_FUSED},\
            Setting optimizer to {IncomingOptimizerNames.ADAMW_ORT_FUSED}")
        args.optim = IncomingOptimizerNames.ADAMW_ORT_FUSED

    if "iou" in args.metric_for_best_model and args.task_name != Tasks.HF_MULTI_LABEL_IMAGE_CLASSIFICATION:
        err_msg = f"{args.metric_for_best_model} metric supported only for {Tasks.HF_MULTI_LABEL_IMAGE_CLASSIFICATION}"
        raise ACFTValidationException._with_error(
            AzureMLError.create(ArgumentInvalid, argument_name="metric_for_best_model",
                                expected_type=err_msg)
        )

    # Prepare args as per the TrainingArguments class+
    use_fp16 = bool(args.precision == 16)

    # Read the default deepspeed config if the apply_deepspeed is set to true without providing config file
    if args.apply_deepspeed and args.deepspeed_config is None:
        args.deepspeed_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zero1.json")
        with open(args.deepspeed_config) as fp:
            ds_dict = json.load(fp)
            use_fp16 = "fp16" in ds_dict and "enabled" in ds_dict["fp16"] and ds_dict["fp16"]["enabled"]

    args.fp16 = use_fp16
    args.deepspeed = args.deepspeed_config if args.apply_deepspeed else None
    if args.metric_for_best_model in MetricConstants.METRIC_LESSER_IS_BETTER:
        args.metric_greater_is_better = False
    else:
        args.metric_greater_is_better = True
    args.load_best_model_at_end = True
    args.report_to = None
    logger.info(f"metric_for_best_model - {args.metric_for_best_model}")
    logger.info(f"metric_greater_is_better - {args.metric_greater_is_better}")

    # setting arguments as needed for the core
    args.model_selector_output = args.model_path
    args.output_dir = SettingParameters.DEFAULT_OUTPUT_DIR

    # TODO: overwriting the save_as_mlflow_model flag to True. Otherwise, it will fail the pipeline service since it
    #  expects the mlflow model folder to create model asset. It can be modified iff outputs of the component can be
    #  optional.
    args.save_as_mlflow_model = True

    # Disable adding prefixes to logger.
    args.set_log_prefix = False
    logger.info(f"Using log prefix - {args.set_log_prefix}")

    logger.info(args)

    # Saving the args is done in `finetune_runner` to handle the distributed training
    finetune_runner(args)


if __name__ == "__main__":
    main()
