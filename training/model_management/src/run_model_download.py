# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.downloader import download_model, ModelSource
from azureml.model.mgmt.utils.common_utils import init_tc, tc_log
from huggingface_hub.hf_api import HfApi, ModelInfo, ModelFilter
from typing import List

HF_ENDPOINT = "https://huggingface.co"


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, help="Model source ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-download-metadata", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    return parser


def check_model_id(model_id):
    """Hugging face model info."""
    try:
        model_list: List[ModelInfo] = HfApi(endpoint=HF_ENDPOINT).list_models(filter=ModelFilter(model_name=model_id))
        for info in model_list:
            if model_id == info.modelId:
                return True
    except Exception as e:
        raise ValueError(f"Failed to validate model id : {e}")
    return False


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

    if not check_model_id(model_id):
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
