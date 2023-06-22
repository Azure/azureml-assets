# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model downloader module."""

import argparse
import json
from azureml.model.mgmt.downloader import download_model, ModelSource
from applicationinsights import TelemetryClient

tc = TelemetryClient("71b954a8-6b7d-43f5-986c-3d3a6605d803")

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
    tc.track_event(name="FM_import_pipeline_debug_logs", 
                       properties={"message":f"Print args"})
    for arg, value in args.__dict__.items():
        tc.track_event(name="FM_import_pipeline_debug_logs",
                          properties={"message":f"{arg} => {value}"})   
        print(f"{arg} => {value}")
    tc.flush()
    if not ModelSource.has_value(model_source):
        tc.track_event(name="FM_import_pipeline_debug_logs",
                            properties={"message":f"Unsupported model source {model_source}"})
        tc.flush()
        raise Exception(f"Unsupported model source {model_source}")
    tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message":f"Downloading model ..."})
    print("Downloading model ...")
    model_download_details = download_model(
        model_source=model_source, model_id=model_id, download_dir=model_output_dir
    )
    tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message":f"Model files downloaded at: {model_output_dir} !!!"})
    print(f"Model files downloaded at: {model_output_dir} !!!")

    with open(model_download_metadata_path, "w") as f:
        json.dump(model_download_details, f)
    tc.track_event(name="FM_import_pipeline_debug_logs", properties={"message":f"Successfully persisted model info !!!"})
    tc.flush()
    print("Successfully persisted model info !!!")
