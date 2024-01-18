# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Flow utils."""

import os
import re
from pathlib import Path
import yaml
import concurrent.futures
import json

from utils.logging_utils import log_debug, log_error, log_warning
from utils.utils import run_command


CONFIG_ROOT = 'scripts/promptflow-ci/test-configs'
CONFIG_FILE = 'test_config.json'


def _assign_flow_values(flow_dirs):
    """Assign the flow values and update flow.dag.yaml."""
    log_debug("\n=======Start overriding values for flows=======")
    for flow_dir in flow_dirs:
        with open(Path(flow_dir) / "flow.dag.yaml", "r") as dag_file:
            flow_dag = yaml.safe_load(dag_file)
        flow_name = flow_dir.parents[0].name
        config_dir = Path(os.path.join(CONFIG_ROOT, flow_name))
        config_file = Path(os.path.join(config_dir, CONFIG_FILE))
        if not (config_dir.exists() and config_file.exists()):
            log_warning(
                f"No available flow values found for '{flow_name}'. Skip flow values assignment.")
        else:
            with open(config_file, "r") as fin:
                values = json.load(fin)
            # Override connection/inputs in nodes
            log_debug(f"Start overriding values for nodes for '{flow_dir}'.")
            if "nodes" in values:
                for override_node in values["nodes"]:
                    for flow_node in flow_dag["nodes"]:
                        if flow_node["name"] == override_node["name"]:
                            if "connection" in override_node:
                                # Override connection
                                flow_node["connection"] = override_node["connection"]
                            if "inputs" in override_node:
                                for input_name, input_value in override_node["inputs"].items():
                                    # Override input
                                    flow_node["inputs"][input_name] = input_value

            with open(flow_dir / "flow.dag.yaml", "w", encoding="utf-8") as dag_file:
                yaml.dump(flow_dag, dag_file, allow_unicode=True)
        if not os.path.exists(Path(flow_dir)/"samples.json"):
            with open(flow_dir/"samples.json", 'w', encoding="utf-8") as sample_file:
                samples = []
                sample = {}
                for key, val in flow_dag["inputs"].items():
                    value = val.get("default")
                    if isinstance(value, list):
                        if not value:
                            value.append("default")
                    elif isinstance(value, str):
                        if value == "":
                            value = "default"
                    sample[key] = value
                samples.append(sample)
                json.dump(sample, sample_file, indent=4)
    log_debug("=======Complete overriding values for flows=======\n")
    return flow_dirs


def _create_run_yamls(flow_dirs):
    """Create run.yml."""
    log_debug("\n=======Start creating run.yaml for flows=======")
    run_yaml = {
        "$schema": "https://azuremlschemas.azureedge.net/promptflow/latest/Run.schema.json",
        "flow": '.',
        "data": 'samples.json'
    }
    for flow_dir in flow_dirs:
        with open(flow_dir / "run.yml", "w", encoding="utf-8") as dag_file:
            yaml.dump(run_yaml, dag_file, allow_unicode=True)
    log_debug("=======Complete creating run.yaml for flows=======\n")
    return


def submit_func(run_path, sub, rg, ws):
    """Worker function to submit flow run."""
    command = f"pfazure run create --file {run_path} --subscription {sub} -g {rg} -w {ws}"
    res = run_command(command)
    res = res.stdout.split('\n')
    return res


def get_run_id_and_url(res, sub, rg, ws):
    """Resolve run_id an url from log."""
    run_id = ""
    portal_url = ""
    for line in res:
        log_debug(line)
        if ('"name":' in line):
            match = re.search(r'"name": "(.*?)",', line)
            if match:
                run_id = match.group(1)
                portal_url = (
                    f"https://ml.azure.com/prompts/flow/bulkrun/run/{run_id}/details"
                    f"?wsid=/subscriptions/{sub}/resourceGroups/{rg}/providers"
                    f"/Microsoft.MachineLearningServices/workspaces/{ws}"
                    )
                log_debug(f"runId: {run_id}")
    return run_id, portal_url


def submit_flow_runs_using_pfazure(flow_dirs, sub, rg, ws):
    """Multi thread submit flow run using pfazure."""
    results = {}
    handled_failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(submit_func, os.path.join(flow_dir, 'run.yml'), sub, rg, ws): flow_dir
            for flow_dir in flow_dirs
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                flow_dir = futures[future]
                res = future.result()
                log_debug(f"Submit test run in flow dir:{flow_dir}.")
                run_id, portal_url = get_run_id_and_url(res, sub, rg, ws)
                results[run_id] = portal_url
            except Exception as exc:
                failure_message = f"Submit test run failed. Flow dir: {flow_dir}.  Error: {exc}."
                log_error(failure_message)
                handled_failures.append(failure_message)
    return results, handled_failures
