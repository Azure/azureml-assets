# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Flow utils."""

import os
import re
from pathlib import Path
import yaml
import concurrent.futures

from utils.logging_utils import log_debug, log_error
from utils.utils import run_command


def _assign_flow_values(flow_dirs):
    """Assign the flow values and update flow.dag.yaml."""
    log_debug("\n=======Start overriding values for flows=======")

    for flow_dir in flow_dirs:
        flow_dir_name = flow_dir.name
        flow_dir_name = flow_dir_name.replace("-", "_")

        with open(Path(flow_dir) / "flow.dag.yaml", "r") as dag_file:
            flow_dag = yaml.safe_load(dag_file)
        # Override connection/inputs in nodes
        log_debug(f"Start overriding values for nodes for '{flow_dir}'.")
        for flow_node in flow_dag["nodes"]:
            if "connection" in flow_node:
                flow_node["connection"] = "aoai_connection"
            if "inputs" in flow_node:
                if "deployment_name" in flow_node["inputs"]:
                    if flow_node["source"].get("tool") == "promptflow.tools.embedding.embedding":
                        flow_node["inputs"]["deployment_name"] = "text-embedding-ada-002"
                    else:
                        flow_node["inputs"]["deployment_name"] = "gpt-35-turbo"
                if "connection" in flow_node["inputs"]:
                    flow_node["inputs"]["connection"] = "aoai_connection"
        with open(flow_dir / "flow.dag.yaml", "w", encoding="utf-8") as dag_file:
            yaml.dump(flow_dag, dag_file, allow_unicode=True)
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
        flow_dir_name = flow_dir.name
        flow_dir_name = flow_dir_name.replace("-", "_")
        with open(flow_dir / "run.yml", "w", encoding="utf-8") as dag_file:
            yaml.dump(run_yaml, dag_file, allow_unicode=True)
    log_debug("=======Complete creating run.yaml for flows=======\n")
    return


def submit_func(run_path, sub, rg, ws):
    """Worker function to submit flow run."""
    command = f"pfazure run create --file  {run_path} --subscription {sub} -g {rg} -w {ws}"
    res = run_command(command)
    res = res.stdout.split('\n')
    return res


def get_run_id_and_url(res):
    """Resolve run_id an url from log."""
    run_id = ""
    portal_url = ""
    for line in res:
        log_debug(line)
        if ('"portal_url":' in line):
            match = re.search(r'/run/(.*?)/details', line)
            if match:
                portal_url = line.strip()
                run_id = match.group(1)
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
                run_id, portal_url = get_run_id_and_url(res)
                results[run_id] = portal_url
            except Exception as exc:
                failure_message = f"Submit test run failed. Flow dir: {flow_dir}.  Error: {exc}."
                log_error(failure_message)
                handled_failures.append(failure_message)
    return results, handled_failures
