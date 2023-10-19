# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
This module link output on run conditionally.

If condition is `true`, link `output` to `input_a`.
If condition is `false`, link `output` to `input_b`.
"""
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str)
parser.add_argument("--input_a_registration_details_folder", type=str)
parser.add_argument("--input_b_registration_details_folder", type=str)
parser.add_argument("--input_a_mlflow_model_folder", type=str)
parser.add_argument("--input_b_mlflow_model_folder", type=str)
parser.add_argument("--output_registration_detail_folder", type=str)
parser.add_argument("--output_mlflow_model_folder", type=str)
args, _ = parser.parse_known_args()
print(f"Condition output component received args: {args}.")

condition = args.condition.lower() == "true"

# gather all files

destination_registration_folder = args.output_registration_detail_folder
destination_mlflow_model_folder = args.output_mlflow_model_folder

if condition:
    print("Copy input_a to output")
    source_model_folder = args.input_a_registration_details_folder
    source_registration_folder = args.input_a_mlflow_model_folder
else:
    print("Copy input_b to output")
    source_model_folder = args.input_b_registration_details_folder
    source_registration_folder = args.input_b_mlflow_model_folder


shutil.copytree(source_model_folder, destination_mlflow_model_folder)
shutil.copytree(source_registration_folder, destination_registration_folder)
