import argparse
import json
from collections import defaultdict
import importlib
import sys

from promptflow.client import load_flow
from azure.ai.evaluation import evaluate
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
import pandas as pd
from utils import get_mlclient, extract_model_info

from logging_utilities import swallow_all_exceptions, get_logger, custom_dimensions, \
    current_run
from azureml.automl.core.shared.logging_utilities import mark_path_as_loggable
import os
import constants

# Mark current path as allowed
mark_path_as_loggable(os.path.dirname(__file__))
custom_dimensions.app_name = constants.TelemetryConstants.COMPONENT_NAME
logger = get_logger(name=__name__)
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
custom_dims_dict = vars(custom_dimensions)


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
    for evaluator in evaluators:
        try:
            if "path" not in evaluator:
                raise ValueError("Path not provided in evaluator config.")
            root_dir = evaluator["path"]
            download_path = f"./{evaluator['name']}"
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
                evaluator["local_path"] = find_file_and_get_parent_dir(download_path)
            else:
                raise ValueError(f"Invalid model asset id: {root_dir}")
        except Exception as e:
            logger.info(f"Error downloading evaluator {evaluator['path']}: {e}")
    return evaluators


def load_evaluators(input_evaluators):
    """Initialize the evaluators using  correct parameters and credentials for rai evaluators."""
    loaded_evaluators, loaded_evaluator_configs = {}, {}
    for evaluator in input_evaluators:
        evaluator_name = evaluator["name"]
        init_params = evaluator.get("init_params", {})
        update_value_in_dict(init_params, "AZURE_OPENAI_API_KEY", lambda x: os.environ[x.upper()])
        flow = load_evaluator(evaluator["local_path"])
        if any(rai_eval in evaluator["path"] for rai_eval in rai_evaluators):
            init_params["credential"] = AzureMLOnBehalfOfCredential()
        loaded_evaluators[evaluator_name] = flow(**init_params)
        loaded_evaluator_configs[evaluator_name] = evaluator["evaluate_config"]
    return loaded_evaluators, loaded_evaluator_configs


def run_evaluation(command_line_args, evaluators, evaluator_configs):
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
    "HateUnfairnessEvaluator",
    "Sexual-Content-Evaluator",
    "Hate-and-Unfairness-Evaluator",
    "Violent-Content-Evaluator",
    "Self-Harm-Related-Content-Evaluator",
]


@swallow_all_exceptions(logger)
def run(args):
    evaluators = json.loads(args["evaluators"])
    evaluators = download_evaluators_and_update_local_path(evaluators)
    evaluators, evaluator_configs = load_evaluators(evaluators)
    run_evaluation(args, evaluators, evaluator_configs)


if __name__ == "__main__":
    args = get_args()
    run(args)