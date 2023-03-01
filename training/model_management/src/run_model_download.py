# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.config import ModelSource, PathType
from azureml.model.mgmt.utils.model_download_utils import download_model


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", required=True, help="Model source ")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-info", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()

    model_source = args.model_source
    model_id = args.model_id
    model_info = args.model_info
    model_output_dir = args.model_output_dir

    print("Print args")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelSource.has_value(model_source) and not PathType.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    if ModelSource.has_value(model_source):
        model_uri_type = ModelSource.get_path_type(model_source)
        model_uri = ModelSource.get_base_url(model_source).format(model_id)

    if PathType.has_value(model_source):
        model_uri_type = model_source
        model_uri = model_id
        model_id = model_uri.split("/")[-1]

    print("Downloading model ...")
    download_details = download_model(model_uri_type, model_uri, model_output_dir)
    print(f"Model files downloaded at: {model_output_dir} !!!")

    # save model info to output json
    model_info_dict = {
        "model_id": model_id,
        "model_name": model_id,
        "model_source": model_source,
        "model_uri": model_uri,
        "model_uri_type": model_uri_type,
        "metadata": {
            "download_details": download_details,
        }
    }

    with open(model_info, "w") as f:
        json.dump(model_info_dict, f)
    print("Successfully persisted model info !!!")
