# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluate for a built-in or custom evulator."""
import argparse
import json
import logging
import mlflow
import pandas as pd
import os
import re
import requests
import shutil
import sys

from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.ai.evaluation import evaluate
from save_evaluation import load_evaluator
from model_target import ModelTarget

AZURE_ENDPOINT = "AzureEndpoint"
API_KEY = "ApiKey"
AZURE_DEPLOYMENT = "AzureDeployment"
TYPE = "Type"
INPUT_EVAL_DATA_DIR = "INPUT_eval_data"
OUTPUT_EVAL_DATA_DIR = "OUTPUT_eval_data"
MODEL_PARAMS = "ModelParams"
MODEL_CONFIG = "ModelConfig"
SYSTEM_MESSAGE = "SystemMessage"
FEW_SHOT_EXAMPLES = "FewShotExamples"
DATA_MAPPING = "dataMapping"
DEFAULT_DATA_MAPPING = {
            "query": "${data.query}",
            "context": "${data.context}",
        }
DATA_MAPPING_REGEX_PATTERN = r'\${data\.(.*?)}'
QUERY_KEY = "query"
CONTEXT_KEY = "context"
RESPONSE_KEY = "response"
GENERATED_RESPONSE_KEY = "generated_response"
GENERATED_RESPONSE_MAPPING = f"${{data.{GENERATED_RESPONSE_KEY}}}"

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
    rai_evaluators = json.loads(command_line_args.rai_evaluators)
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
    evaluators_o = json.loads(command_line_args.evaluators)
    for evaluator_name, evaluator in evaluators_o.items():
        if evaluator["DataMapping"]:
            evaluator_config[evaluator_name] = {"column_mapping": evaluator["DataMapping"]}
    return evaluator_config


def create_model_target_and_data_mapping(command_line_args):
    """Get model target configuration from user input."""
    if not command_line_args.eval_target:
        logger.info("No eval_target provided. Returning None.")
        return (None, None)
    try:
        logger.info("Eval_target provided.")
        target_config = json.loads(command_line_args.eval_target)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in eval_target: {e}")

    model_config = target_config.get(MODEL_CONFIG, {})

    api_key_env = model_config.get(API_KEY, "")
    api_key_value = os.environ.get(api_key_env.upper(), "")
    if not api_key_value:
        raise RuntimeError(f"API key environment variable '{api_key_env.upper()}' is missing or empty!")

    model_config_type = str(model_config.get(TYPE, ""))
    logger.info(f"  - Type: {model_config_type}")

    endpoint = model_config.get(AZURE_ENDPOINT, "")
    model_params = target_config.get(MODEL_PARAMS, {}).copy()
    data_mapping = model_params.pop(DATA_MAPPING, None)
    if data_mapping is None:
        data_mapping = DEFAULT_DATA_MAPPING
        logger.info(f"Using default dataMapping: {data_mapping}")

    system_message = target_config.get(SYSTEM_MESSAGE, "")
    few_shot_examples = target_config.get(FEW_SHOT_EXAMPLES, [])

    logger.info("Creating ModelTarget with values:")
    logger.info(f"  - Endpoint: {endpoint}")
    logger.info(f"  - ApiKey: {'[HIDDEN]' if api_key_value else 'MISSING'}")
    logger.info(f"  - ModelParams: {model_params}")
    logger.info(f"  - SystemMessage: {system_message}")
    logger.info(f"  - FewShotExamples: {few_shot_examples}")
    logger.info(f"  - DataMapping: {data_mapping}")

    model_target = ModelTarget(
        endpoint=endpoint,
        api_key=api_key_value,
        model_params=model_params,
        system_message=system_message,
        few_shot_examples=few_shot_examples,
    )

    return (model_target, data_mapping)


def apply_target_on_data(data, model_target, data_mapping):
    """Apply target on input data."""
    input_filename = os.path.basename(data)
    name, ext = os.path.splitext(input_filename)
    df = pd.DataFrame()
    try:
        if ext == ".csv":
            df = pd.read_csv(data)
        elif ext == ".jsonl":
            df = pd.read_json(data, lines=True)
        else:
            raise RuntimeError(f"Supported file types: jsonl and csv. {ext} not supported.")
    except Exception as e:
        raise RuntimeError(f"Failed to read file '{data}': {e}")

    mapping = {}
    for key, value in data_mapping.items():
        # ignore mappings with empty values
        if value == "":
            continue
        match = re.search(DATA_MAPPING_REGEX_PATTERN, value)
        if match:
            mapping[key] = match.group(1)
        else:
            logger.info(
                f'Data mapping did not match the format "{key}":"${{data.<colName in dataset>}}", '
                f'using given {value}'
            )
            mapping[key] = value

    if QUERY_KEY not in mapping:
        logger.info("Using default mapping for query column")
        mapping[QUERY_KEY] = QUERY_KEY
    if CONTEXT_KEY not in mapping:
        logger.info("Using default mapping for context column")
        mapping[CONTEXT_KEY] = CONTEXT_KEY

    for index, row_object in df.iterrows():
        row = row_object.to_dict()
        mapped_row = {}
        for actual_col, input_data_col in mapping.items():
            if input_data_col in row:
                mapped_row[actual_col] = row[input_data_col]
            elif actual_col == QUERY_KEY:
                raise ValueError(
                    f"'{QUERY_KEY}' is missing in the data_mapping. "
                    f"The {QUERY_KEY} column mapping is required."
                )

        query = mapped_row.get(QUERY_KEY, "")
        context = mapped_row.get(CONTEXT_KEY, "")
        response = model_target.generate_response(query=query, context=context)
        df.loc[index, GENERATED_RESPONSE_KEY] = response

    output_dir = data.replace(INPUT_EVAL_DATA_DIR, OUTPUT_EVAL_DATA_DIR)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    output_filename = ""
    if ext == ".csv":
        output_filename = f"{name}_output.csv"
    else: # .jsonl
        output_filename = f"{name}_output.jsonl"

    output_file = os.path.join(os.path.dirname(output_dir), output_filename)
    if ext == ".csv":
        df.to_csv(output_file, index=False)
    else:
        df.to_json(output_file, orient="records", lines=True)
    logger.info(f"Saved updated DataFrame to {output_file}")

    return output_file


def update_evaluator_config_mapping_for_generated_response(command_line_args, evaluator_config):
    """Ensure 'response' key exists in 'column_mapping' and update it."""
    for evaluator_name, config in evaluator_config.items():
        if "column_mapping" not in config:
            config["column_mapping"] = {}
        evaluator_config[evaluator_name]["column_mapping"][RESPONSE_KEY] = GENERATED_RESPONSE_MAPPING

    # create generated_repsonse mapping for evaluators without column_mapping provided.
    evaluators_o = json.loads(command_line_args.evaluators)
    for evaluator_name in evaluators_o:
        if evaluator_name not in evaluator_config:
            evaluator_config[evaluator_name] = {"column_mapping": {}}
        evaluator_config[evaluator_name]["column_mapping"][RESPONSE_KEY] = GENERATED_RESPONSE_MAPPING

    return evaluator_config


def run_evaluation(command_line_args, evaluators, evaluator_config, model_target, data_mapping):
    """Run evaluation using evaluators."""
    logger.info(f"Running the evaluators: {list(evaluators.keys())}")
    logger.info(f"With the model target {model_target} and dataMapping {data_mapping}")

    data = command_line_args.eval_data
    logger.info(f"Evaluation Data: {data}")
    if model_target:
        logger.info("Applying target on data")
        data = apply_target_on_data(data=data, model_target=model_target, data_mapping=data_mapping)
        logger.info(f"Evaluation Data after applying target: {data}")
        logger.info("Updating evaluator config for generated_response data mapping")
        evaluator_config = update_evaluator_config_mapping_for_generated_response(command_line_args, evaluator_config)

    logger.info(f"With the evaluator config {evaluator_config}")
    results = evaluate(
        data=data,
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
parser.add_argument("--rai_evaluators", type=str, help="Comma-separated list of RAI evaluators", required=False)
parser.add_argument("--eval_target", type=str, help="Optional evaluation target", required=False)

args = parser.parse_args()

if __name__ == '__main__':
    copy_evaluator_files(args)
    evaluators = initialize_evaluators(args)
    evaluator_config = get_evaluator_config(args)
    model_target, data_mapping = create_model_target_and_data_mapping(args)
    logger.info("*************** Collecting Result of Evaluators ******************")
    # Run the evaluation
    with mlflow.start_run() as run:
        try:
            run_evaluation(args, evaluators, evaluator_config, model_target, data_mapping)
        except Exception as e:
            logger.error("EXCEPT", e)
            get_promptflow_run_logs()
            raise RuntimeError(f"EXCEPT: {e}")
