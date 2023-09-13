# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing function for model selector component."""

import argparse

from azureml.acft.image import VERSION, PROJECT_NAME
from azureml.acft.image.components.finetune.factory.mappings import MODEL_FAMILY_CLS
from azureml.acft.image.components.model_selector.component import ImageModelSelector
from azureml.acft.image.components.common.constants import LOGS_TO_BE_FILTERED_IN_APPINSIGHTS

from azureml.acft.common_components import get_logger_app, set_logging_parameters, LoggingLiterals
from azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator import swallow_all_exceptions

logger = get_logger_app("azureml.acft.image.scripts.components.model_selector.model_selector")


def get_common_parser():
    """
    Add arguments and returns the parser. Here we add all the arguments for model selector.

    Those arguments that are not relevant for the input task should be ignored.
    """
    parser = argparse.ArgumentParser(
        description="Model selector for image models", allow_abbrev=False
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Folder to store model selector outputs",
        default="model_selector_output",
    )
    # # Model Family
    parser.add_argument(
        "--model_family",
        type=str,
        choices=(
            MODEL_FAMILY_CLS.HUGGING_FACE_IMAGE,
            MODEL_FAMILY_CLS.MMDETECTION_IMAGE
        ),
        required=True,
        help="Which framework the model belongs to. E.g. HuggingFaceImage"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="Name of the model.",
    )

    # Continual Finetuning
    parser.add_argument(
        "--pytorch_model",
        default=None,
        type=str,
        help="Input Asset Id of pytorch model",
    )
    parser.add_argument(
        "--mlflow_model",
        default=None,
        type=str,
        help="Input Asset Id of pytorch model"
    )

    return parser


@swallow_all_exceptions(time_delay=60)
def main():
    """Driver function for model selector."""
    # common args
    common_parser = get_common_parser()
    common_args, _ = common_parser.parse_known_args()

    args = argparse.Namespace(**vars(common_args))

    set_logging_parameters(
        task_type="model_selector-" + args.model_family,
        acft_custom_dimensions={
            LoggingLiterals.PROJECT_NAME: PROJECT_NAME,
            LoggingLiterals.PROJECT_VERSION_NUMBER: VERSION,
        },
        azureml_pkg_denylist_logging_patterns=LOGS_TO_BE_FILTERED_IN_APPINSIGHTS,
    )

    logger.info("Calling model downloader from common component package to download models")

    ImageModelSelector(
        pytorch_model=args.pytorch_model,
        mlflow_model=args.mlflow_model,
        model_name=args.model_name,
        model_family=args.model_family,
        output_dir=args.output_dir,
    ).run_workflow()
    logger.info("Model download successful!")


if __name__ == "__main__":
    main()
