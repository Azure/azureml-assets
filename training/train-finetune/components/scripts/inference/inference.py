# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for inference component.
"""

import os
import argparse
import json

from azureml.train.finetune.core.constants import task_definitions
from azureml.train.finetune.core.constants.constants import SaveFileConstants
from azureml.train.finetune.core.utils.logging_utils import get_logger_app
from azureml.train.finetune.core.drivers.inference import run_inference

logger = get_logger_app()


def get_task_parser(task_metadata):
    """Parse all arguments."""
    parser = argparse.ArgumentParser(task_metadata.hf_task_name, allow_abbrev=False)
    for item in task_metadata.arg_parse_data:
        arg_name = item["dest"]
        # NOTE force converting the target_key required status to False
        # This is done only for inference as for inference the target_key might
        # might not exist
        if arg_name == task_metadata.dataset_target_key:
            item["required"] = False
        key = f"--{arg_name}"
        parser.add_argument(key, **item)

    return parser


def get_common_parser():
    """
    Gets the common parser object.
    """
    parser = argparse.ArgumentParser(description="Inference component", allow_abbrev=False)

    # NOTE Task, Model and Lora parameters are read from finetuning args
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
        default="./ds_config_zero3.json",
        help="If apply_deepspeed is set to True, this file will be used as deepspeed config",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank passed by torch distributed launch",
    )

    # Evaluation settings
    parser.add_argument("--batch_size", default=4, type=int, help="Test batch size")
    parser.add_argument(
        "--metric_name",
        default=None,
        type=str,
        help="Metric name in hugging face dataset",
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

    # Model folder
    # containing the model, tokenizer files
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="output folder of finetuning containing the model, config and tokenizer files",
    )

    parser.add_argument(
        "--test_file_path",
        default=None,
        type=str,
        help="valid file name in dataset_path folder",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        type=str,
        help="Output dir to save the finetune model and other metadata",
    )

    return parser


if __name__ == "__main__":
    # PREPROCESSING ARGS
    # common args
    common_parser = get_common_parser()
    common_args, _ = common_parser.parse_known_args()
    # reading the args already specified in finetuning here
    # task_name, model_name_or_path, lora_args, tokenizer settings
    finetune_args_path = os.path.join(common_args.model_path, SaveFileConstants.FinetuneArgsSavePath)
    with open(finetune_args_path, "r") as rptr:
        finetune_args = json.load(rptr)
        # adding the finetune arguments
        common_args.task_name = finetune_args.get("task_name")
        common_args.model_name_or_path = finetune_args.get("model_name_or_path")
        common_args.apply_lora = finetune_args.get("apply_lora")
        common_args.lora_alpha = finetune_args.get("lora_alpha")
        common_args.lora_dropout = finetune_args.get("lora_dropout")
        common_args.lora_r = finetune_args.get("lora_r")
        common_args.merge_lora_weights = finetune_args.get("merge_lora_weights")

        # check if lora layers to be added to the model
        common_args.add_lora_layers = (not common_args.merge_lora_weights) and common_args.apply_lora

    # decode user task name to HF task name
    common_args.user_task_name = common_args.task_name
    task_metadata = getattr(task_definitions, common_args.user_task_name)()
    common_args.hf_task_name = task_metadata.hf_task_name
    common_args.hf_problem_type = task_metadata.hf_problem_type

    if common_args.hf_task_name is None:
        raise ValueError(f"Unsupported Task: {common_args.user_task_name}")

    # add task related arguments
    task_parser = get_task_parser(task_metadata)
    task_args, _ = task_parser.parse_known_args()

    # combine common args and task related args
    args = argparse.Namespace(**vars(common_args), **vars(task_args))

    args.test_data_path = args.test_file_path

    # Check if the label_separator is not empty
    # This is applicable only for MultiLabelClassification
    if getattr(args, "label_separator", None) and len(args.label_separator) == 0:
        raise ValueError("Label separator cannot be an empty string")

    # Convert the boolean variables from string to bool
    if isinstance(args.apply_ort, str):
        args.apply_ort = args.apply_ort.lower() == "true"
    if isinstance(args.apply_deepspeed, str):
        args.apply_deepspeed = args.apply_deepspeed.lower() == "true"

    if args.apply_deepspeed and args.deepspeed_config is None:
        args.deepspeed_config = "./ds_config_zero3.json"

    args.use_fp16 = (args.precision == 16)

    logger.info(f"Lora Enabled: {args.apply_lora}")
    logger.info(f"ORT Enabled: {args.apply_ort}")
    logger.info(f"DeepSpeed Enabled: {args.apply_deepspeed}")
    logger.info(f"FP16 Enabled: {args.use_fp16}")

    # update the task info
    args.dataset_target_key = getattr(args, task_metadata.dataset_target_key)

    # update the task info
    decode_dataset_columns = []
    decode_datast_columns_dtypes = []
    # if_target_key_exist flag decides whether or not to do the label key formatting
    # if True ==> data format happens => metrics will be computed on model predictions
    # if False ==> ONLY model prediction happens
    args.if_target_key_exist = False
    for idx, var in enumerate(task_metadata.dataset_columns):
        decoded_arg = getattr(args, var, None)  # This will translate to, for example, prompt = args.sentence1_key
        if decoded_arg is not None:
            if var == task_metadata.dataset_target_key:
                args.if_target_key_exist = True
            decode_dataset_columns.append(decoded_arg)
            decode_datast_columns_dtypes.append(task_metadata.dataset_columns_dtypes[idx])
    args.keep_columns = decode_dataset_columns
    args.keep_columns_dtypes = decode_datast_columns_dtypes

    # INFERENCE ARGS
    # get the num_labels
    class_names_path = os.path.join(args.model_path, SaveFileConstants.ClassesSavePath)
    with open(class_names_path, "r") as rptr:
        class_names = json.load(rptr)[SaveFileConstants.ClassesSaveKey]
        logger.info("Class names : {}".format(class_names))
        args.num_labels = len(class_names)
    logger.info("Args : {}".format(args))

    run_inference(args)
