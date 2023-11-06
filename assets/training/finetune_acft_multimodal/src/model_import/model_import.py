# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for model selector component.
"""

from pathlib import Path
import argparse
from argparse import Namespace

from azureml.acft.multimodal.components import PROJECT_NAME, VERSION
from azureml.acft.multimodal.components.constants.constants import Tasks
from azureml.acft.multimodal.components.task_factory import get_task_runner

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals


logger = get_logger_app("azureml.acft.multimodal.components.scripts.components.model_selector.model_selector")


def get_parser():
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

    parser.add_argument(
        "--data_modalities",
        default="text-image",
        type=str,
        help="Modalities to consider",
    )

    # Task settings
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Model id used to load model checkpoint.",
    )

    parser.add_argument(
        "--task_name",
        type=str,
        default=Tasks.MUTIMODAL_CLASSIFICATION,
        help="Task Type.",
    )

    # Continual Finetuning
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        help="input folder path containing pytorch model for further finetuning"
    )
    parser.add_argument(
        "--mlflow_model_path",
        default=None,
        type=str,
        help="input folder path containing mlflow model for further finetuning"
    )

    return parser


def model_selector(args: Namespace):
    """
    main function handling model selector
    """
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.huggingface_id is not None:
        args.model_name = args.huggingface_id
    else:
        # TODO Revist whether `model_id` is still relevant
        args.model_name = args.model_name_or_path

    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_modelselector(**vars(args))


if __name__ == "__main__":
    #args
    parser = get_parser()
    args, _ = parser.parse_known_args()

    set_logging_parameters(
        task_type=Tasks.MUTIMODAL_CLASSIFICATION,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        }
    )

    model_selector(args)
