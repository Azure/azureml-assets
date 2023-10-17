# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
This module downloads the model from the registry.
"""
import argparse, json, os
from azureml.model.mgmt.config import ComponentVariables
from azureml.model.mgmt.utils.common_utils import get_mlclient

parser = argparse.ArgumentParser()
parser.add_argument("--registration_details_folder", type=str)
parser.add_argument("--mlflow_model_folder", type=str)

args, _ = parser.parse_known_args()
print(f"Condition output component received args: {args}.")

with open(os.path.join(args.registration_details_folder, ComponentVariables.REGISTRATION_DETAILS_JSON_FILE)) as f:
    registration_details = json.load(f)

model_id = registration_details["id"]
model = registration_details["name"]
version = registration_details["version"]
registry = model_id.split("/")[3]
print(f"model_id is {model_id} and registry is {registry} and version is {version} and model is {model}")
ml_client_registry = get_mlclient(registry_name=registry)

mlflow_model_from_registry = ml_client_registry.models.download(
    name=model, version=version, download_path=args.mlflow_model_folder)

print("Downloaded model from registry.")
