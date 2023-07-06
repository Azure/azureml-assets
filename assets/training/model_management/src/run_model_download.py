# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.config import AppName
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.exceptions import swallow_all_exceptions
from azureml.model.mgmt.utils.logging_utils import custom_dimensions, get_logger


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.DOWNLOAD_MODEL


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, help="Model source ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-download-metadata", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    return parser


@swallow_all_exceptions(logger)
def run():
    """Run model download."""
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()

    model_source = args.model_source
    model_id = args.model_id
    model_download_metadata_path = args.model_download_metadata
    model_output_dir = args.model_output_dir

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    logger.info(f"Model source: {model_source}")
    logger.info(f"Model id: {model_id}")

    logger.info("Downloading model")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir
    )

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)

    logger.info("Download completed.")


if __name__ == "__main__":
    run()
