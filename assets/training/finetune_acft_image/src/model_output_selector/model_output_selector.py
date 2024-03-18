# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
File containing function for model output selector component for images.

Return output for runtime image command component if condition is true and output for FT component otherwise.

TODO: Replace with control flow https://msdata.visualstudio.com/Vienna/_workitems/edit/2306663.
"""

from azureml.core.run import Run

import argparse
import shutil

from azureml.acft.common_components import (
    get_logger_app, set_logging_parameters, LoggingLiterals, PROJECT_NAME, VERSION
)

logger = get_logger_app(
    "azureml.acft.common_components.scripts.components.model_output_selector.model_output_selector"
)


def get_common_parser():
    """Get common parser for model output selector."""
    parser = argparse.ArgumentParser(
        description="Output model selector for image models", allow_abbrev=False
    )

    # Input MLFlow models
    parser.add_argument(
        "--mlflow_model_t",
        type=str,
        required=False,
        help="Input MLFlow model for true block."
    )
    parser.add_argument(
        "--mlflow_model_f",
        type=str,
        required=False,
        help="Input MLFLow model for false block."
    )

    # Input Pytorch models
    parser.add_argument(
        "--pytorch_model_t",
        type=str,
        required=False,
        help="Input Pytorch model for true block."
    )
    parser.add_argument(
        "--pytorch_model_f",
        type=str,
        required=False,
        help="Input Pytorch model for false block."
    )

    # Condition based on which outputs will be selected.
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Condition based on which output models will be selected."
    )

    # Output models
    parser.add_argument(
        "--output_mlflow",
        type=str,
        help="MLFLow output model selected based on given condition.",
    )

    parser.add_argument(
        "--output_pytorch",
        type=str,
        help="Pytorch output model selected based on given condition.",
    )

    return parser


def copy_model(model_t: str, model_f: str, model_output: str, condition: bool):
    """Copy model based on condition.

    :param model_t: Path to model to be copied if condition is true.
    :type model_t: str
    :param model_f: Path to model to be copied if condition is false.
    :type model_f: str
    :param model_output: Path to output model.
    :type model_output: str
    :param condition: Condition based on which model will be copied.
    :type condition: bool
    """
    logger.info(f"Selecting model based on condition: {condition}")
    try:
        if condition:
            shutil.copytree(model_t, model_output, dirs_exist_ok=True)
        else:
            shutil.copytree(model_f, model_output, dirs_exist_ok=True)
    except Exception as e:
        raise ValueError("Exception in downloading model: {}".format(e))


if __name__ == "__main__":

    # common args
    common_parser = get_common_parser()
    common_args, _ = common_parser.parse_known_args()

    args = argparse.Namespace(**vars(common_args))

    set_logging_parameters(
        task_type="model_output_selector",
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        },
    )

    # For now, we need to read the file. Will skip this once we can take a dependency on
    # conditional_output. TODO: Task 2306663 (https://msdata.visualstudio.com/Vienna/_workitems/edit/2306663)
    f = open(args.condition, "r")
    if f.read() in ["true", "True"]:
        cond = True
    else:
        cond = False

    copy_model(args.mlflow_model_t, args.mlflow_model_f, args.output_mlflow, cond)
    copy_model(args.pytorch_model_t, args.pytorch_model_f, args.output_pytorch, cond)

    # Copy the model to parent output folder as well
    # in same format as command job run
    azureml_run = Run.get_context()
    parent_run = azureml_run.parent

    if parent_run:
        logger.info(f"Saving model to outputs of {parent_run}")
        if cond:
            parent_run.upload_folder(name='outputs', path=args.pytorch_model_t)
            parent_run.upload_folder(name='outputs/mlflow-model', path=args.mlflow_model_t)
        else:
            parent_run.upload_folder(name='outputs', path=args.pytorch_model_f)
            parent_run.upload_folder(name='outputs/mlflow-model', path=args.mlflow_model_f)
