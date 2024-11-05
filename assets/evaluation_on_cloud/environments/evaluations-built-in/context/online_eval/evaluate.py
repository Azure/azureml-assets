# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main script for the evaluate context."""
import argparse
import json
import logging
from collections import defaultdict
import importlib
import sys
import shutil

from promptflow.client import load_flow
from azure.ai.evaluation import evaluate
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
import pandas as pd
from utils import get_mlclient, extract_model_info

import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_args():
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Inputs

    parser.add_argument("--preprocessed_data", type=str, dest="preprocessed_data",
                        default="./preprocessed_data_output.jsonl")
    parser.add_argument("--evaluated_data", type=str, dest="evaluated_data", default="./evaluated_data_output.jsonl")
    parser.add_argument("--evaluators", type=str, dest="evaluators")
    parser.add_argument("--sampling_rate", type=str, dest="sampling_rate", default="1")

    args, _ = parser.parse_known_args()
    return vars(args)


def load_evaluator(evaluator):
    """Load the evaluator from the given path."""
    logger.info(f"Loading evaluator {evaluator}")
    loaded_evaluator = load_flow(evaluator)
    logger.info(loaded_evaluator)
    logger.info(
        f"Loading module {os.getcwd()} {loaded_evaluator.entry.split(':')[0]} from {loaded_evaluator.path.parent.name}"
    )
    module_path = os.path.join(
        loaded_evaluator.path.parent, loaded_evaluator.entry.split(":")[0] + ".py"
    )
    module_name = loaded_evaluator.entry.split(":")[0]
    logger.info(f"Loading module {module_name} from {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    logger.info(f"Loaded module {mod}")
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    eval_class = getattr(mod, loaded_evaluator.entry.split(":")[1])
    return eval_class


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
            return os.path.abspath(dirpath)


# Todo: We should not load evaluators every time the component runs
def download_evaluators_and_update_local_path(evaluators):
    """Find the flex flow or any given file in a directory and return the parent directory."""
    for evaluator_name, evaluator in evaluators.items():
        try:
            root_dir = evaluator["Id"]
            download_path = f"./{evaluator_name}"
            if root_dir.startswith("azureml://"):
                model_info = extract_model_info(root_dir)
                if model_info is None:
                    logger.info(f"Invalid model asset id: {root_dir}")
                    return
                if model_info['type'] == "workspace_registered":
                    mlclient = get_mlclient()
                elif model_info['type'] == "registry_registered":
                    mlclient = get_mlclient(registry_name=model_info['registry'])
                mlclient.models.download(name=model_info["model_name"], version=model_info["version"],
                                         download_path=download_path)
                evaluators[evaluator_name]["local_path"] = find_file_and_get_parent_dir(download_path)
            else:
                raise ValueError(f"Invalid model asset id: {root_dir}")
        except Exception as e:
            logger.info(f"Error downloading evaluator {evaluator['Id']}: {e}")
    return evaluators


def copy_evaluator_files(command_line_args):
    """Copy the mounted evaluator files to the relative paths to enable read/write."""
    evaluators = json.loads(command_line_args["evaluators"])
    evaluator_name_id_map = json.loads(command_line_args.evaluator_name_id_map)
    for evaluator_name, evaluator_id in evaluator_name_id_map.items():
        dir_path = find_file_and_get_parent_dir(evaluator_id)
        if dir_path:
            shutil.copytree(dir_path, f"./{evaluator_name}")
            logger.info(f"Copying {dir_path} to ./{evaluator_name}")
            copied_dir = os.listdir(f"./{evaluator_name}")
            logger.info(f"Directory ./{evaluator_name} now contains: {copied_dir}")
            sys.path.append(os.path.abspath(f"./{evaluator_name}"))
            evaluators[evaluator_name]["local_path"] = os.path.abspath(f"./{evaluator_name}")
        else:
            logger.info(f"Directory for evaluator {evaluator_name} not found.")
    return evaluators


def load_evaluators(input_evaluators):
    """Initialize the evaluators using  correct parameters and credentials for rai evaluators."""
    loaded_evaluators, loaded_evaluator_configs = {}, {}
    for evaluator_name, evaluator in input_evaluators.items():
        init_params = evaluator.get("InitParams", {})
        update_value_in_dict(init_params, "AZURE_OPENAI_API_KEY", lambda x: os.environ[x.upper()])
        flow = load_evaluator(evaluator["local_path"])
        if any(rai_eval in evaluator["Id"] for rai_eval in rai_evaluators):
            init_params["credential"] = AzureMLOnBehalfOfCredential()
        loaded_evaluators[evaluator_name] = flow(**init_params)
        loaded_evaluator_configs[evaluator_name] = {"column_mapping": evaluator.get("DataMapping", {})}
    return loaded_evaluators, loaded_evaluator_configs


def run_evaluation(command_line_args, evaluators, evaluator_configs):
    """Run the evaluation."""
    # Todo: can we get only results back instead of the whole response?
    results = evaluate(data=command_line_args["preprocessed_data"], evaluators=evaluators,
                       evaluator_config=evaluator_configs)
    logger.info("Evaluation Completed")
    logger.info("results here", results)
    final_results = defaultdict(list)
    for result in results["rows"]:
        for evaluator_name in evaluators:
            result_key = f"outputs.{evaluator_name}"
            filtered_result = {k: v for k, v in result.items() if k.startswith(result_key)}
            if len(filtered_result) == 1:
                final_results[evaluator_name].append(filtered_result[list(filtered_result.keys())[0]])
            else:
                logger.info(f"Found multiple results for {evaluator_name}. Adding as json string.")
                final_results[evaluator_name].append(json.dumps(filtered_result))
    final_results = pd.DataFrame(final_results)
    logger.info(final_results)
    final_results.to_json(command_line_args["evaluated_data"], orient="records", lines=True)


rai_evaluators = [
    "Sexual-Content-Evaluator",
    "Hate-and-Unfairness-Evaluator",
    "Violent-Content-Evaluator",
    "Self-Harm-Related-Content-Evaluator",
    "Groundedness-Pro-Evaluator",
    "Protected-Material-Evaluator",
    "Indirect-Attack-Evaluator",
]


def run(args):
    """Entry point of model prediction script."""
    evaluators = copy_evaluator_files(args)
    evaluators, evaluator_configs = load_evaluators(evaluators)
    run_evaluation(args, evaluators, evaluator_configs)


if __name__ == "__main__":
    args = get_args()
    run(args)
