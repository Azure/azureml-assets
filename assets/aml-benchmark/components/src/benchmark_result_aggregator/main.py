# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, Optional, Tuple, cast
import argparse
import mlflow
from azureml.core import Run

from utils.logging import get_logger, log_mlflow_params
from utils.io import read_json_data, save_json_to_file
from utils.exceptions import swallow_all_exceptions
from utils.aml_run_utils import (
    get_all_runs_in_current_experiment,
    get_compute_information,
    get_parent_run_id,
    get_step_name,
    get_mlflow_model_name_version
)


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


def _get_pipeline_params() -> Tuple[
    Dict[str, Dict[str, Any]], Dict[str, Any], Optional[str], Optional[str], Optional[str]
]:
    """Get the details of the pipeline."""
    runs = get_all_runs_in_current_experiment()
    current_run = Run.get_context()
    pipeline_params: Dict[str, Dict[str, Any]] = {}
    loggable_params: Dict[str, Any] = {}
    model_name = None
    model_version = None
    model_registry = None
    for run in runs:
        run_info = run.info
        run_id = run_info.run_id
        run_v1 = Run(experiment=current_run.experiment, run_id=run_id)
        step_name = get_step_name(run)
        run_details = cast(Dict[str, Any], run_v1.get_details())
        logger.info(f"Run details for {run_id}: {run_details}")
        run_definition = run_details.get('runDefinition', None)
        run_v1_properties = run_details.get('properties', None)
        logger.info(f"Run definition for {run_id}: {run_definition}")

        step_params: Dict[str, Any] = {
            'inputs': {},
            'param': {},
            'run_id': run_id,
            'start_time': run_info.start_time,
            'end_time': run_info.end_time,
            'status': run_info.status,
            'maxRunDurationSeconds': run_details.get('maxRunDurationSeconds', None),
            'is_reused': False,
        }

        if run_v1_properties.get('azureml.isreused', None) == 'true':
            step_params['is_reused'] = True
            step_params['reused_run_id'] = \
                run_v1_properties.get('azureml.reusedrunid', None)
            step_params['reused_pipeline_runid'] = \
                run_v1_properties.get('azureml.reusedpipelinerunid', None)


        if run_definition is not None:

            # Log input assets
            input_assets = run_definition.get('inputAssets', None)
            if input_assets is not None:
                for key, value in input_assets.items():
                    step_params['inputs'][key] = {
                        'assetId': value['asset']['assetId'],
                        'type': value['asset']['type'],
                        'mechanism': value['mechanism'],
                    }
                    if step_params['inputs'][key]['type'] == 'MLFlowModel':
                        if model_name is None:
                            # We found a mlflow model in the pipeline
                            model_uri: str = step_params['inputs'][key]['assetId']
                            model_name, model_version, model_registry = \
                                get_mlflow_model_name_version(model_uri)
                        else:
                            # If an mlflow model is found multiple times,
                            # we don't know which one was benchmarked.
                            # Hence, handle ambiguous case by not logging model.
                            model_name = None
                            model_version = None
                            model_registry = None

            # Log node params
            node_params = run_definition.get('environmentVariables', None)
            if node_params is not None:
                for key, value in node_params.items():
                    AZUREML_PARAM_PREFIX = 'AZUREML_PARAMETER_'
                    if key.startswith(AZUREML_PARAM_PREFIX):
                        key = key[len(AZUREML_PARAM_PREFIX):]
                    step_params['param'][key] = value
                    loggable_params[f"{step_name}.param.{key}"] = value

            # Log environment definition
            env_definition = run_definition.get('environment', None)
            if env_definition is not None:
                step_params['environment_asset_id'] = env_definition['assetId']
                step_params['environment_version'] = env_definition['version']
                step_params['environment_name'] = env_definition['name']

            # Log component id
            step_params['component_id'] = \
                run_definition['componentConfiguration'].get('componentIdentifier', None)

            # Log node count
            step_params['node_count'] = run_definition['nodeCount']
            loggable_params[f"{step_name}.node_count"] = run_definition['nodeCount']


        # Get compute information
        log_files = run_details.get('logFiles', None)
        vm_size = get_compute_information(log_files, run_v1)
        step_params['vm_size'] = vm_size
        loggable_params[f"{step_name}.vm_size"] = vm_size

        logger.info(step_params)
        pipeline_params[step_name] = step_params
    return pipeline_params, loggable_params, model_name, model_version, model_registry


def _log_params_and_metrics(parameters: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Log mlflow params and metrics to current run and parent run."""
    filtered_metrics = {}
    for key in metrics:
        if not isinstance(metrics[key], dict):
            filtered_metrics[key] = metrics[key]
    # Log to current run
    log_mlflow_params(**parameters)
    mlflow.log_metrics(filtered_metrics)
    # Log to parent run
    parent_run_id = get_parent_run_id()
    ml_client = mlflow.tracking.MlflowClient()
    for param_name, param_value in parameters.items():
        param_value_to_log = param_value
        if isinstance(param_value, str) and len(param_value) > 500:
            param_value_to_log = param_value[: 497] + '...'
        ml_client.log_param(parent_run_id, param_name, param_value_to_log)
    for metric_name, metric_value in filtered_metrics.items():
        ml_client.log_metric(parent_run_id, metric_name, metric_value)

@swallow_all_exceptions(logger)
def main(
    quality_metrics_path: Optional[str],
    performance_metrics_path: Optional[str],
    output_dataset_path: str,
):
    """The main function for the benchmark result aggregator."""
    quality_metrics = read_json_data(quality_metrics_path)
    for key in quality_metrics.keys():
        value = quality_metrics[key]

    performance_metrics = read_json_data(performance_metrics_path)
    parameters: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    runs = get_all_runs_in_current_experiment()
    for run in runs:
        step_name = get_step_name(run)
        params: Dict[str, Any] = run.data.params
        metrics_for_run: Dict[str, Any] = run.data.metrics
        # Aggregate mlflow params
        for key, value in params.items():
            param_name = f"{step_name}.{key}"
            if param_name in parameters:
                logger.warning(msg=f"Replacing param {param_name}.")
            parameters[f"{step_name}.{key}"] = value
        # Aggregate mlflow metrics
        for key, value in metrics_for_run.items():
            metric_name = f"{step_name}.{key}"
            if metric_name in metrics:
                logger.warning(msg=f"Replacing metric {metric_name}.")
            metrics[f"{step_name}.{key}"] = value
    (pipeline_params,
    loggable_pipeline_params,
    model_name,
    model_version,
    model_registry)= \
        _get_pipeline_params()
    result: Dict[str, Any] = {
        'run_id': get_parent_run_id(),
        'quality_metrics': quality_metrics,
        'performance_metrics': performance_metrics,
        'mlflow_parameters': parameters,
        'mlflow_metrics': metrics,
        'simplified_pipeline_params': loggable_pipeline_params,
        'pipeline_params': pipeline_params,
    }
    if model_name is not None:
        result['model_name'] = model_name
        loggable_pipeline_params['model_name'] = model_name
    if model_version is not None:
        result['model_version'] = model_version
        loggable_pipeline_params['model_version'] = model_version
    if model_registry is not None:
        result['model_registry'] = model_registry
        loggable_pipeline_params['model_registry'] = model_registry
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
