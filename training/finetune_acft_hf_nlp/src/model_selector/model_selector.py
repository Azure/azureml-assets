# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""File containing function for model selector component."""
from pathlib import Path
import argparse
from argparse import Namespace

from azureml.acft.contrib.hf.nlp.task_factory import get_task_runner

from azureml.acft.accelerator.utils.decorators import swallow_all_exceptions
from azureml.acft.accelerator.utils.logging_utils import get_logger_app


logger = get_logger_app()


def get_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for all the tasks.

    Those arguments that are not relevant for the input task should be ignored.
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
        "--model_id",
        type=str,
        default=None,
        help="Model id used to load model checkpoint.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="SingleLabelClassification",
        help="Task Name",
    )

    # Continual Finetuning
    parser.add_argument(
        "--pytorch_model_path",
        default=None,
        type=str,
        help="model path containing pytorch model"
    )
    parser.add_argument(
        "--mlflow_model_path",
        default=None,
        type=str,
        help="model path containing mlflow model"
    )

    return parser


def model_selector(args: Namespace):
    """Model selector."""
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.huggingface_id is not None:
        args.model_name = args.huggingface_id
    else:
        # TODO Revist whether `model_id` is still relevant
        args.model_name = args.model_id

    task_runner = get_task_runner(task_name=args.task_name)()
    task_runner.run_modelselector(**vars(args))


@swallow_all_exceptions(logger)
def main():
    """Parse args and import model."""
    # args
    parser = get_parser()
    args, _ = parser.parse_known_args()

    model_selector(args)


if __name__ == "__main__":
    main()
