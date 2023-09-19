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
parser.add_argument("--input_a", type=str)
parser.add_argument("--input_b", type=str)
parser.add_argument("--condition", type=str)
args, _ = parser.parse_known_args()
print(f"Condition output component received args: {args}.")
if args.input_a is None and args.input_b is None:
    raise Exception("Got 'input_a' and 'input_b' both be None.")

run = Run.get_context()
workspace = run.experiment.workspace
condition = args.condition.lower() == "true"
try:
    if condition:
        print("Linking output to input_a ...")
        dataset = Dataset.get_by_id(workspace, id=args.input_a)
    else:
        print("Linking output to input_b ...")
        dataset = Dataset.get_by_id(workspace, id=args.input_b)
    run.output_datasets["output"].link(dataset)
except Exception as e:
    raise Exception(f"Link output failed with {e}.")