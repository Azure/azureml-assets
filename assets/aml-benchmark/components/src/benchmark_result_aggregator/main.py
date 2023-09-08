# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Any, Dict, List, Optional, Tuple
import argparse
import os
import sys
import tempfile
import re

import mlflow
from mlflow.entities import Run as MLFlowRun
from azureml.core import Run

sys.path.append(os.getcwd())

from utils.helper import get_logger, log_mlflow_params
from utils.io import read_json_data, save_json_to_file


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the arguments for the component."""
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--quality_metrics_path",
        type=str,
        default=None,
        help="Path to the quality metrics in json format.",
    )
    parser.add_argument(
        "--performance_metrics_path",
        type=str,
        default=None,
        help="Path to the performance metrics in json format.",
    )
    parser.add_argument(
        "--output_dataset_path",
        type=str,
        help="The json file with all of the aggregated results.",
    )

    arguments, _ = parser.parse_known_args()
    logger.info(f"Arguments: {arguments}")
    return arguments


def _get_experiment_name() -> str:
    return Run.get_context().experiment.name

def _get_parent_run_id() -> str:
    return Run.get_context().parent.id


def _get_all_runs_in_current_experiment() -> List[MLFlowRun]:
    experiment_name = _get_experiment_name()
    parent_run_id = _get_parent_run_id()
    runs: List[MLFlowRun] = mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'",
        output_format='list'
    )
    return [
        run for run in runs
        if run.info.run_id != Run.get_context().id and run.info.run_id != parent_run_id
    ]


def _getattr_or_none(obj: Dict[Any, Any], key: Any) -> Any:
    return obj[key] if key in obj else None


def _get_pipeline_params() -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    runs = _get_all_runs_in_current_experiment()
    current_run = Run.get_context()
    pipeline_params: Dict[str, Dict[str, Any]] = {}
    loggable_params: Dict[str, Any] = {}
    for run in runs:
        step_params = {}
        run_info = run.info
        run_id = run_info.run_id
        run_v1 = Run(experiment=current_run.experiment, run_id=run_id)
        step_name = _get_step_name(run)
        run_details = run_v1.get_details()
        logger.info(run_details)
        run_definition = _getattr_or_none(run_details, 'runDefinition')
        logger.info(run_definition)
        if run_definition is not None:

            # Log input assets
            input_assets = _getattr_or_none(run_definition, 'inputAssets')
            if input_assets is not None:
                for key, value in input_assets.items():
                    step_params[f"inputs.{key}"] = value

            # Log output assets
            output_assets = _getattr_or_none(run_definition, 'outputData')
            if output_assets is not None:
                for key, value in output_assets.items():
                    step_params[f"outputs.{key}"] = value

            # Log node params
            node_params = _getattr_or_none(run_definition, 'environmentVariables')
            if node_params is not None:
                for key, value in node_params.items():
                    AZUREML_PARAM_PREFIX = 'AZUREML_PARAMETER_'
                    if key.startswith(AZUREML_PARAM_PREFIX):
                        key = key[len(AZUREML_PARAM_PREFIX):]
                    step_params[f"param.{key}"] = value
                    loggable_params[f"{step_name}.param.{key}"] = value

            # Log environment definition
            env_definition = _getattr_or_none(run_definition, 'environment')
            if env_definition is not None:
                step_params['environment_asset_id'] = env_definition['assetId']
                step_params['environment_version'] = env_definition['version']
                step_params['environment_name'] = env_definition['name']

            # Log component id
            step_params['component_id'] = run_definition['componentConfiguration']

            # Log node count
            step_params['node_count'] = run_definition['nodeCount']
            loggable_params[f"{step_name}.node_count"] = run_definition['nodeCount']

        # Log run start/end time and status
        step_params['start_time'] = run_info.start_time
        step_params['end_time'] = run_info.end_time
        step_params['status'] = run_info.status

        # Get compute information
        log_files = _getattr_or_none(run_details, 'logFiles')
        if log_files is not None:
            file_name = 'execution-wrapper.log'
            complete_file_name = 'system_logs/lifecycler/' + file_name
            if complete_file_name in log_files:
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        run_v1.download_file(complete_file_name, tmp)
                        with open(os.path.join(tmp, file_name), 'r') as file:
                            content = file.read()
                            compute_match = re.search(
                                r'vm_size: Some\("([^\"]*)\"',
                                content,
                            )
                    if compute_match:
                        step_params['vm_size'] = compute_match.group(1)
                        loggable_params[f"{step_name}.vm_size"] = compute_match.group(1)
                except Exception as ex:
                    logger.warn(f"Failed to get system_logs/lifecycler/execution-wrapper.log \
                                for run {run_id}")
        print(step_params)
        logger.info(step_params)
        pipeline_params[step_name] = step_params
    return pipeline_params, loggable_params


def _get_step_name(run: MLFlowRun) -> str:
    stepName = 'stepName'
    if stepName in run.data.tags:
        return run.data.tags[stepName]
    return run.info.run_name


def _log_params_and_metrics(parameters: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    filtered_metrics = {}
    for key in metrics:
        if not isinstance(metrics[key], dict):
            filtered_metrics[key] = metrics[key]
    mlflow.log_params(parameters)
    mlflow.log_metrics(filtered_metrics)
    parent_run_id = _get_parent_run_id()
    ml_client = mlflow.tracking.MlflowClient()
    for param_name, param_value in parameters.items():
        ml_client.log_param(parent_run_id, param_name, param_value)
    for metric_name, metric_value in filtered_metrics.items():
        ml_client.log_metric(parent_run_id, metric_name, metric_value)

def main(
    quality_metrics_path: Optional[str],
    performance_metrics_path: Optional[str],
    output_dataset_path: str,
):
    quality_metrics = read_json_data(quality_metrics_path)
    for key in quality_metrics.keys():
        value = quality_metrics[key]

    performance_metrics = read_json_data(performance_metrics_path)
    parameters: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    runs = _get_all_runs_in_current_experiment()
    for run in runs:
        step_name = _get_step_name(run)
        params: Dict[str, Any] = run.data.params
        metrics_for_run: Dict[str, Any] = run.data.metrics
        for key, value in params.items():
            param_name = f"{step_name}.{key}"
            if param_name in parameters:
                logger.warn(msg=f"Replacing param {param_name}.")
            parameters[f"{step_name}.{key}"] = value
        for key, value in metrics_for_run.items():
            metric_name = f"{step_name}.{key}"
            if metric_name in metrics:
                logger.warn(msg=f"Replacing metric {metric_name}.")
            metrics[f"{step_name}.{key}"] = value
    pipeline_params, loggable_pipeline_params = _get_pipeline_params()
    result: Dict[str, Dict[str, Any]] = {
        'quality_metrics': quality_metrics,
        'performance_metrics': performance_metrics,
        'mlflow_parameters': parameters,
        'pipeline_params': pipeline_params,
        'mlflow_metrics': metrics,
    }
    save_json_to_file(result, output_dataset_path)
    _log_params_and_metrics(
        parameters={**parameters, **loggable_pipeline_params},
        metrics={**quality_metrics, **performance_metrics},
    )


if __name__ == "__main__":
    args = parse_args()
    main(
        quality_metrics_path=args.quality_metrics_path,
        performance_metrics_path=args.performance_metrics_path,
        output_dataset_path=args.output_dataset_path,
    )
