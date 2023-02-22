# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.config import ModelType, PathType
from azureml.model.mgmt.utils.model_download_utils import download_model


def _get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-uri", required=True, help="A public URI referencing the model.")
    parser.add_argument(
        "--model-uri-type",
        help="Supported storage containers to download model from.",
        required=True,
    )
    parser.add_argument(
        "--model-type",
        default="custom_model",
        required=False,
        help="Model type supported by AzureML viz. custom/mlflow/triton. **custom** will be assumed as default, if unspecified.",
    )
    parser.add_argument("--model-info", required=True, help="Model source info file path")
    parser.add_argument("--model-output-dir", required=True, help="Model download directory")
    return parser


if __name__ == "__main__":
    parser = _get_parser()
    args, unknown_args_ = parser.parse_known_args()

    model_uri = args.model_uri
    model_uri_type = args.model_uri_type
    model_type = args.model_type
    model_info = args.model_info
    model_output_dir = args.model_output_dir

    print("Print args")
    for arg, value in args.__dict__.items():
        print(f"{arg} => {value}")

    if not ModelType.has_value(model_type):
        raise Exception(f"Unsupported model type {model_type}")
    if not PathType.has_value(model_uri_type):
        raise Exception(f"Unsupported model download URI type {model_uri_type}")

    print("Downloading model ...")
    download_details = download_model(model_uri_type, model_uri, model_output_dir)
    print(f"Model downloaded to the mounted location {model_output_dir} !!!")

    # save model info to output json
    model_info_dict = {
        "model_uri": model_uri,
        "model_uri_type": model_uri_type,
        "model_type": model_type,
        "metadata": {
            "download_details": download_details,
        }
    }

    with open(model_info, "w") as f:
        json.dump(model_info_dict, f)
    print("Successfully persisted model info !!!")
