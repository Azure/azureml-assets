# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.common_utils import init_tc, tc_log, check_model_id


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, help="Model source ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-download-metadata", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()
    init_tc()
    model_source = args.model_source
    model_id = args.model_id
    model_download_metadata_path = args.model_download_metadata
    model_output_dir = args.model_output_dir

    tc_log("Print args")

    if not ModelSource.has_value(model_source):
        tc_log("Unsupported model source")
        raise Exception(f"Unsupported model source {model_source}")

    if model_source == ModelSource.HUGGING_FACE and not check_model_id(model_id):
        tc_log("Model id is not valid")
        raise Exception(f"Model id {model_id} is not valid")

    tc_log(f"Model source: {model_source}")
    tc_log(f"Model id: {model_id}")

    tc_log("Downloading model ...")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir
    )

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)

    tc_log("Successfully persisted model info ")
