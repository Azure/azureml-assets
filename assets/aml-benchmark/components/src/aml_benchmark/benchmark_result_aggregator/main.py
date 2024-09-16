# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Entry script for Benchmark Result Aggregator Component."""

from typing import Any, Dict, Optional, Tuple, cast
import argparse
from azureml.core import Run
import json
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.io import read_json_data, save_json_to_file
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.constants import DATASET_CONFIG_2_NAME_MAP, REQUIRED_TELEMETRY_KEYS_MAP
from aml_benchmark.utils.aml_run_utils import (
    get_all_runs_in_current_experiment,
    get_compute_information,
    get_parent_run_id,
    get_step_name,
    get_mlflow_model_name_version,
    is_model_prediction_component_present,
    get_evaluation_type
)


logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
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


def _get_run_reused_properties(run_v1_properties: Any) -> Dict[str, Any]:
    """
    Return properties relate to run reused from previous pipeline runs status.

    :param run_v1_properties: Run properties using SDK v1.

    :returns: The properties related to run reused status.
    """
    if run_v1_properties.get("azureml.isreused", None) == "true":
        return {
            "is_reused": True,
            "reused_run_id": run_v1_properties.get("azureml.reusedrunid", None),
            "reused_pipeline_runid": run_v1_properties.get(
                "azureml.reusedpipelinerunid", None
            ),
        }
    return {
        "is_reused": False,
    }


def _get_run_environment_properties(
    run_definition: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Get the environment details for a run.

    :param run_definition: The run definition for which environment details have to be fetched.

    :returns: The environment details.
    """
    if run_definition is None:
        return {}
    env_definition = run_definition.get("environment", None)
    if env_definition is not None:
        return {
            "environment_asset_id": env_definition["assetId"],
            "environment_version": env_definition["version"],
            "environment_name": env_definition["name"],
        }
    return {}


def _get_run_params(run_definition: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get the input parameters for the run.

    :param run_definition: The run definition for which input params have to be fetched.

    :returns: The run's input parameters.
    """
    if run_definition is None:
        return {}
    node_params = run_definition.get("environmentVariables", None)
    params: dict[str, Any] = {}
    if node_params is not None:
        for param_name, param_value in node_params.items():
            AZUREML_PARAM_PREFIX = "AZUREML_PARAMETER_"
            if param_name.startswith(AZUREML_PARAM_PREFIX):
                param_name = param_name[len(AZUREML_PARAM_PREFIX):]
            params[param_name] = param_value
    return params


def _get_input_assets(run_definition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get input assets for a run.

    :param run_definition: The run definition for which input assets have to be fetched.

    :returns: The run's input assets.
    """
    input_assets = run_definition.get("inputAssets", None)
    input_assets_dict: Dict[str, Any] = {}
    if input_assets is not None:
        for input_name, asset_details in input_assets.items():
            try:
                asset_info = asset_details.get("asset", None)
                if asset_info is None:
                    asset_id = None
                    asset_type = None
                else:
                    asset_id = asset_info.get("assetId", None)
                    asset_type = asset_info.get("type", None)
                input_assets_dict[input_name] = {
                    "assetId": asset_id,
                    "type": asset_type,
                    "mechanism": asset_details.get("mechanism", None),
                }
            except Exception as ex:
                logger.warning(
                    f"Falied to get asset information for {input_name}: {asset_details} due to {ex}"
                )
    return input_assets_dict


def _get_pipeline_params() -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Any],
    Optional[str],
    Optional[str],
    Optional[str],
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
        run_id = run.info.run_id
        run_v1 = Run(experiment=current_run.experiment, run_id=run_id)
        step_name = get_step_name(run)
        run_details = cast(Dict[str, Any], run_v1.get_details())
        run_definition = run_details.get("runDefinition", {})
        run_v1_properties = run_details.get("properties", {})

        # Get run parameters, input assets and compute_information.
        run_params = _get_run_params(run_definition)
        input_assets = _get_input_assets(run_definition)
        log_files = run_details.get("logFiles", None)
        vm_size = get_compute_information(log_files, run_v1)
        component_config = run_definition.get("componentConfiguration", {})

        pipeline_params[step_name] = {
            "inputs": input_assets,
            "param": run_params,
            "run_id": run_id,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "status": run.info.status,
            "node_count": run_definition.get("nodeCount", None),
            "vm_size": vm_size,
            "maxRunDurationSeconds": run_details.get("maxRunDurationSeconds", None),
            "component_id": component_config.get("componentIdentifier", None),
            **_get_run_reused_properties(run_v1_properties),
            **_get_run_environment_properties(run_definition),
        }

        # Update loggable parameters.
        loggable_params[f"{step_name}.node_count"] = run_definition.get(
            "nodeCount", None
        )
        loggable_params[f"{step_name}.vm_size"] = vm_size
        for run_param_key, run_param_value in run_params.items():
            loggable_params[f"{step_name}.param.{run_param_key}"] = run_param_value

        # Get the model details in the pipeline.
        for asset in input_assets.values():
            if asset["type"] != "MLFlowModel":
                continue
            model_uri = asset["assetId"]
            name, version, registry = get_mlflow_model_name_version(model_uri)
            if model_name is None:
                model_name = name
                model_version = version
                model_registry = registry
            elif (
                model_name != name
                or model_version != version
                or model_registry != registry
            ):
                model_name = model_version = model_registry = None

    return pipeline_params, loggable_params, model_name, model_version, model_registry


@swallow_all_exceptions(logger)
def main(
    quality_metrics_path: Optional[str],
    performance_metrics_path: Optional[str],
    output_dataset_path: str,
) -> None:
    """
    Entry function for the benchmark result aggregator.

    :param quality_metrics_path: The path to quality metrics file/directory.
    :param performance_metrics_path: The path to the performance metrics file/directory.
    :param output_dataset_path: The path to file where the result has to be stored.
    """
    quality_metrics = read_json_data(quality_metrics_path)
    performance_metrics = read_json_data(performance_metrics_path)

    parameters: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    runs = get_all_runs_in_current_experiment()
    for run in runs:
        step_name = get_step_name(run)
        params: Dict[str, Any] = run.data.params
        metrics_for_run: Dict[str, Any] = run.data.metrics
        # Aggregate mlflow params
        for param_key, param_value in params.items():
            param_name = f"{step_name}.{param_key}"
            if param_name in parameters:
                logger.warning(msg=f"Replacing param {param_name}.")
            parameters[f"{step_name}.{param_key}"] = param_value
        # Aggregate mlflow metrics
        for param_key, metric_value in metrics_for_run.items():
            metric_name = f"{step_name}.{param_key}"
            if metric_name in metrics:
                logger.warning(msg=f"Replacing metric {metric_name}.")
            metrics[f"{step_name}.{param_key}"] = metric_value
    (
        pipeline_params,
        loggable_pipeline_params,
        model_name,
        model_version,
        model_registry,
    ) = _get_pipeline_params()
    result: Dict[str, Any] = {
        "run_id": get_parent_run_id(),
        "quality_metrics": quality_metrics,
        "performance_metrics": performance_metrics,
        "mlflow_parameters": parameters,
        "mlflow_metrics": metrics,
        "simplified_pipeline_params": loggable_pipeline_params,
        "pipeline_params": pipeline_params,
    }
    if model_name is not None:
        result["model_name"] = model_name
        loggable_pipeline_params["model_name"] = model_name
    if model_version is not None:
        result["model_version"] = model_version
        loggable_pipeline_params["model_version"] = model_version
    if model_registry is not None:
        result["model_registry"] = model_registry
        loggable_pipeline_params["model_registry"] = model_registry
    # Collect the telemetry parameters.
    telemetry_details = {"parameters": {"model_as_an_asset": is_model_prediction_component_present(),
                                        "evaluation_type": get_evaluation_type()},
                         "metrics": {}}
    mlflow_parameters = result.get("mlflow_parameters", {})
    for key, telemetry_key in REQUIRED_TELEMETRY_KEYS_MAP.items():
        telemetry_details["parameters"][telemetry_key] = mlflow_parameters.get(key, None)
        if key == "downloader.dataset_name":
            dataset_name = telemetry_details["parameters"][telemetry_key]
            if dataset_name is None:
                config_name = mlflow_parameters.get("downloader.configuration", None)
                telemetry_details["parameters"][telemetry_key] = DATASET_CONFIG_2_NAME_MAP.get(config_name, None)
    for metric_key in ["mlflow_metrics", "quality_metrics"]:
        for k, v in result.get(metric_key, {}).items():
            # All those values which are float or int or string
            if isinstance(v, (int, float, str)):
                try:
                    key = k.split(".", 1)[1]
                except Exception:
                    key = k
                telemetry_details["metrics"][key] = v
    task_name = result.get("simplified_pipeline_params", {}).get("quality.param.task", None)
    telemetry_details["parameters"]["task_name"] = task_name
    logger.info(f"Telemetry details: {json.dumps(telemetry_details)}")
    save_json_to_file(result, output_dataset_path)


if __name__ == "__main__":
    args = _parse_args()
    main(
        quality_metrics_path=args.quality_metrics_path,
        performance_metrics_path=args.performance_metrics_path,
        output_dataset_path=args.output_dataset_path,
    )
