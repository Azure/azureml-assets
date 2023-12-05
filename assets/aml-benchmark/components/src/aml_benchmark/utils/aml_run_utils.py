# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Helper functions for AML runs."""
from typing import List, Optional, Tuple, cast
import os
import tempfile
import re

import mlflow
from mlflow.entities import Run as MLFlowRun
from azureml.core import Run
from aml_benchmark.utils.logging import get_logger

from azureml._common._error_definition.azureml_error import AzureMLError
from .exceptions import BenchmarkValidationException
from .error_definitions import BenchmarkValidationError


logger = get_logger(__name__)


def get_experiment_name() -> str:
    """Get the current experiment name."""
    return Run.get_context().experiment.name


def get_parent_run_id() -> str:
    """Get the run id of the parent of the current run."""
    return cast(str, Run.get_context().parent.id)


def get_all_runs_in_current_experiment() -> List[MLFlowRun]:
    """
    Get a list of all of the runs in the current experiment \
    that are a direct child of the root run except the current run.

    :returns: The list of runs in the current experiment.
    """
    experiment_name = get_experiment_name()
    parent_run_id = get_parent_run_id()
    runs = cast(List[MLFlowRun], mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=f"tags.mlflow.parentRunId='{parent_run_id}'",
        output_format='list'
    ))
    return [
        run for run in runs
        if run.info.run_id != Run.get_context().id and run.info.run_id != parent_run_id
    ]


def get_compute_information(log_files: Optional[List[str]], run_v1: Run) -> Optional[str]:
    """Get the VMType used for a given run."""
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
                    return compute_match.group(1)
            except Exception as ex:
                logger.warn(f"Failed to get system_logs/lifecycler/execution-wrapper.log due to {ex}")
                return None


def get_step_name(run: MLFlowRun) -> str:
    """Get the step name of a given run."""
    stepName = 'stepName'
    if stepName in run.data.tags:
        return run.data.tags[stepName]
    return run.info.run_name


def get_mlflow_model_name_version(model_uri: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Get the mlflow model name and version from a URI.

    :param model_uri: the URI from which the model name and version has to be parsed.

    :returns: The model name and version.
    """
    # If the model is from a registry, parse the information from URI
    if model_uri.startswith("azureml://registries/"):
        model_name = model_uri.split('/')[-3]
        model_version = model_uri.split('/')[-1]
        model_registry = model_uri.split('/')[3]
    else:
        model_name = model_uri
        model_version = None
        model_registry = None
    return model_name, model_version, model_registry


def str2bool(v):
    """Convert str to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details='Boolean value expected.')
        )


def get_run_name(run: Run) -> str:
    """Get the run name."""
    if run.display_name:
        return run.display_name
    if run.name:
        return run.name
    return run.get_details().get("properties", {}).get("azureml.moduleName")


def get_root_run() -> Run:
    """Get the root run."""
    current_run = Run.get_context()
    pipeline_run = current_run
    while pipeline_run.parent is not None:
        pipeline_run = pipeline_run.parent
        logger.info(f"using upper level runs {pipeline_run.id}")
    return pipeline_run


def get_dependent_run(dependent_step) -> Optional[Run]:
    """Get the dependent run."""
    pipeline_run = get_root_run()
    logger.info(f"Checking pipeline run {pipeline_run.id}.")
    for run in pipeline_run.get_children():
        run_name = get_run_name(run)
        logger.info(f"Checking run {run.id} with module_name {run_name}.")
        if run_name == dependent_step:
            return run
    return None


def get_current_step_raw_input_value(input_asset_name: str) -> Optional[str]:
    """Get the input asset."""
    current_run = Run.get_context()
    step = get_root_run().get_details()['runDefinition']['Jobs'].get(current_run.display_name, {})
    return step.get('inputs', {}).get(input_asset_name, {}).get('value')
