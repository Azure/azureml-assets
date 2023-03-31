# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.downloader import download_model, ModelSource


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

    model_source = args.model_source
    model_id = args.model_id
    model_download_metadata_path = args.model_download_metadata
    model_output_dir = args.model_output_dir

    print("Print args")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelSource.has_value(model_source):
        raise Exception(f"Unsupported model source {model_source}")

    print("Downloading model ...")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir)
    print(f"Model files downloaded at: {model_output_dir} !!!")

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)
    print("Successfully persisted model info !!!")
