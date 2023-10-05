# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
This module link output on run conditionally.

If condition is `true`, link `output` to `input_a`.
If condition is `false`, link `output` to `input_b`.
"""
import argparse
from azureml.core import Dataset, Run

parser = argparse.ArgumentParser()
parser.add_argument("--condition", type=str)
parser.add_argument("--input_a_registration_details_folder", type=str)
parser.add_argument("--input_b_registration_details_folder", type=str)
parser.add_argument("--input_a_mlflow_model_folder", type=str)
parser.add_argument("--input_b_mlflow_model_folder", type=str)
args, _ = parser.parse_known_args()
print(f"Condition output component received args: {args}.")

condition = args.condition.lower() == "true"
run = Run.get_context()
if condition:
    print("Copy input_a to output")
    registration_details_folder = Dataset.File.from_files(path=args.input_a_registration_details_folder)
    mlflow_model_folder = Dataset.File.from_files(path=args.input_a_mlflow_model_folder)
else:
    print("Copy input_b to output")
    registration_details_folder = Dataset.File.from_files(path=args.input_b_registration_details_folder)
    mlflow_model_folder = Dataset.File.from_files(path=args.input_b_mlflow_model_folder)

run.output_datasets["registration_details_folder"].link(registration_details_folder)
run.output_datasets["mlflow_model_folder"].link(mlflow_model_folder)