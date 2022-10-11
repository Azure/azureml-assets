# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
File containing function for model selector component.
"""

from importlib import import_module
from azureml.train.finetune.core.drivers.model_selector import model_downloader
import argparse
from azureml.train.finetune.core.utils.logging_utils import get_logger_app
from azureml.train.finetune.core.constants.constants import SaveFileConstants
import os
import json

logger = get_logger_app()


def get_common_parser():
    """
    Adds arguments and returns the parser. Here we add all the arguments for all the tasks.
    Those arguments that are not relevant for the input task should be ignored
    """
    parser = argparse.ArgumentParser(description="Model selector for hugging face models", allow_abbrev=False)

    parser.add_argument(
        "--output_dir",
        default="model_selector_output",
        type=str,
        help="folder to store model selector outputs",
    )

    parser.add_argument(
        "--huggingface_id",
        default=None,
        type=str,
        help="Input HuggingFace model id takes priority over model_id.",
    )

    # Task settings
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="bert-base-uncased",
        help="Model id used to load model checkpoint.",
    )

    # Continual Finetuning
    parser.add_argument(
        "--continual_finetuning_model_path",
        default=None,
        type=str,
        help="input folder path containing model for further finetuning"
    )

    return parser


if __name__ == "__main__":
    # common args
    common_parser = get_common_parser()
    common_args, _ = common_parser.parse_known_args()

    # combine common args and task related args
    args = argparse.Namespace(**vars(common_args))

    if args.huggingface_id:
        args.model_name_or_path = args.huggingface_id

    logger.info("Calling model downloader from driver script to download models")
    model_downloader(args)

    model_selector_args_save_path = os.path.join(args.output_dir, SaveFileConstants.ModelSelectorArgsSavePath)
    logger.info(f"Saving the model selector args to {model_selector_args_save_path}")
    with open(model_selector_args_save_path, "w") as wptr:
        json.dump(vars(args), wptr, indent=2)
