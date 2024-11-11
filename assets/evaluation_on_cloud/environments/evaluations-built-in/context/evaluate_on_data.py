# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluate for a built-in or custom evulator."""
import argparse
import json
import logging
import mlflow
import pandas as pd
import os
import requests
import shutil
import sys

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.evaluation import evaluate
from save_evaluation import load_evaluator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def update_value_in_dict(d, key_substring, new_func):
    """Recursively search for a value containing 'key_substring' and apply 'new_func' to modify it."""
    for key, value in d.items():
        if isinstance(value, dict):
            update_value_in_dict(value, key_substring, new_func)
        elif isinstance(value, str) and key_substring in value:
            d[key] = new_func(value)


def find_file_and_get_parent_dir(root_dir, file_name="flow.flex.yaml"):
    """Find the flex flow or any given file in a directory and return the parent directory."""
    for dirpath, _, filenames in os.walk(root_dir):
        if file_name in filenames:
            logger.info(f"Found {file_name} in {dirpath}")
            return dirpath


def copy_evaluator_files(command_line_args):
    """Copy the mounted evaluator files to the relative paths to enable read/write."""
    evaluator_name_id_map = json.loads(command_line_args.evaluator_name_id_map)
    for evaluator_name, evaluator_id in evaluator_name_id_map.items():
        dir_path = find_file_and_get_parent_dir(evaluator_id)
        if dir_path:
            shutil.copytree(dir_path, f"./{evaluator_name}")
            logger.info(f"Copying {dir_path} to ./{evaluator_name}")
            copied_dir = os.listdir(f"./{evaluator_name}")
            logger.info(f"Directory ./{evaluator_name} now contains: {copied_dir}")
            sys.path.append(os.path.abspath(f"./{evaluator_name}"))
        else:
            logger.info(f"Directory for evaluator {evaluator_name} not found.")


def initialize_evaluators(command_line_args):
    """Initialize the evaluators using  correct parameters and credentials for rai evaluators."""
    evaluators = {}
    evaluators_o = json.loads(command_line_args.evaluators)
    for evaluator_name, evaluator in evaluators_o.items():
        init_params = evaluator["InitParams"]
        update_value_in_dict(init_params, "AZURE_OPENAI_API_KEY", lambda x: os.environ[x.upper()])
        flow = load_evaluator("./" + evaluator_name)
        if any(rai_eval in evaluator["Id"] for rai_eval in rai_evaluators):
            init_params["credential"] = AzureMLOnBehalfOfCredential()
        evaluators[evaluator_name] = flow(**init_params)
    return evaluators


def get_evaluator_config(command_line_args):
    """Get evaluator configuration from user input."""
    evaluator_config = {}
    data_mapping = {}
    evaluators_o = json.loads(command_line_args.evaluators)
    for evaluator_name, evaluator in evaluators_o.items():
        if evaluator["DataMapping"]:
            data_mapping["column_mapping"] = evaluator["DataMapping"]
            evaluator_config[evaluator_name] = data_mapping
    return evaluator_config


def run_evaluation(command_line_args, evaluators, evaluator_config):
    """Run evaluation using evaluators."""
    logger.info(f"Running the evaluators: {list(evaluators.keys())}")
    logger.info(f"With the evaluator config {evaluator_config}")
    results = evaluate(
        data=command_line_args.eval_data,
        evaluators=evaluators,
        evaluator_config=evaluator_config if evaluator_config else None,
    )
    metrics = {}
    for metric_name, metric_value in results["metrics"].items():
        logger.info(f"Logging metric added with name {metric_name}, and value {metric_value}")
        metrics[metric_name] = metric_value
    mlflow.log_metrics(metrics)

    if results and results.get("rows"):
        # Convert the results to a DataFrame
        df = pd.DataFrame(results["rows"])

        # Save the DataFrame as a JSONL file
        df.to_json("instance_results.jsonl", orient="records", lines=True)
        mlflow.log_artifact("instance_results.jsonl")


def get_promptflow_run_logs():
    """Get promptflow run logs."""
    if os.path.exists("/root/.promptflow/.runs/"):
        runs = os.listdir("/root/.promptflow/.runs/")
        for run in runs:
            if os.path.exists(f"/root/.promptflow/.runs/{run}/logs.txt"):
                with open(f"/root/.promptflow/.runs/{run}/logs.txt", "r") as f:
                    logger.info(f"RUN {run} =========================")
                    logger.info(f.read())
    else:
        logger.info("RUN DOES NOT EXIST")


# Create a session for making HTTP requests
session = requests.Session()

# Parse command line arguments and debug to ensure working
parser = argparse.ArgumentParser("eval")
parser.add_argument("--eval_data", type=str)
parser.add_argument("--eval_output", type=str)
parser.add_argument("--evaluators", type=str)
parser.add_argument("--evaluator_name_id_map", type=str)

args = parser.parse_args()
rai_evaluators = ['HateUnfairnessEvaluator', 'Sexual-Content-Evaluator', 'Hate-and-Unfairness-Evaluator',
                  'Violent-Content-Evaluator', 'Self-Harm-Related-Content-Evaluator']

if __name__ == '__main__':
    copy_evaluator_files(args)
    evaluators = initialize_evaluators(args)
    evaluator_config = get_evaluator_config(args)
    logger.info("*************** Collecting Result of Evaluators ******************")
    # Run the evaluation
    with mlflow.start_run() as run:
        try:
            run_evaluation(args, evaluators, evaluator_config)
        except Exception as e:
            logger.error("EXCEPT", e)
            get_promptflow_run_logs()
