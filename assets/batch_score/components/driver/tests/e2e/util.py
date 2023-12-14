# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end test utilities."""

import errno
import os
import shutil
import tempfile

import pytest
import yaml
from azure.ai.ml import MLClient, load_job
from azure.ai.ml.entities import Job
from pydantic.utils import deep_update


def _submit_job_and_monitor_till_completion(
    ml_client: MLClient,
    pipeline_filepath: str,
    yaml_overrides: "list[dict]" = None,
):
    job = _submit_batch_score_job(
        ml_client=pytest.ml_client,
        pipeline_filepath=pipeline_filepath,
        yaml_overrides=yaml_overrides)

    assert job.status == "Preparing"

    try:
        job = _wait_until_termination(ml_client=pytest.ml_client, pipeline_job=job)

        assert job.status == "Completed"
    except Exception:
        raise Exception(f"Exception raised while waiting for termination. Studio url: {job.studio_url}")


def _submit_batch_score_job(ml_client: MLClient,
                            pipeline_filepath: str,
                            yaml_overrides: "list[dict]" = None) -> Job:
    """Register the specified component from local yaml, and then creates a pipeline job from local yaml."""
    """Return the AzureML Job when it reaches terminal state."""
    tmpDir = tempfile.TemporaryDirectory()

    try:
        tmpFile = os.path.join(tmpDir.name, "pipeline.yml")
        create_copy(pipeline_filepath, tmpFile)

        print(f"Copied {pipeline_filepath} to {tmpFile}")
        pipeline_filepath = tmpFile

        # update input pipeline yml values if needed
        if yaml_overrides:
            for yaml_override in yaml_overrides:
                _update_yaml(filename=pipeline_filepath, yaml_override=yaml_override)

        # create pipeline job from yaml
        pipeline_from_yaml = load_job(source=pipeline_filepath)
        pipeline_job = ml_client.create_or_update(pipeline_from_yaml)
        print(f"Pipeline job {pipeline_job.name} created.")

        return pipeline_job
    except Exception:
        raise
    finally:
        tmpDir.cleanup()


def _wait_until_termination(ml_client: MLClient,
                            pipeline_job: Job):
    # stream until the job reaches terminal state
    ml_client.jobs.stream(pipeline_job.name)

    # get the terminal Job object
    job = ml_client.jobs.get(pipeline_job.name)

    return job


def _update_yaml(filename: str, yaml_override: dict):
    """Update a value in a yaml file."""
    with open(filename) as f:
        yml: dict = yaml.safe_load(f)

    yml = deep_update(yml, yaml_override)
    print(f"Updated keys with yaml_override: {yaml_override}")

    with open(filename, 'w') as f:
        yaml.safe_dump(yml, f, default_flow_style=False)
    print(f"Final file: {yml}")


def _get_component_name(component_filepath: str) -> "tuple[str, str]":
    with open(component_filepath) as f:
        yml = yaml.safe_load(f)

    if 'name' in yml:
        yml['name'] = f"{yml['name']}_devops_test"

    return yml['name']


def _set_and_get_component_name_ver(component_filepath: str, component_version: str) -> "tuple[str, str]":
    with open(component_filepath) as f:
        yml = yaml.safe_load(f)

    if 'name' in yml:
        yml['name'] = _get_component_name(component_filepath)

    if 'version' in yml:
        yml['version'] = component_version

    with open(component_filepath, 'w') as f:
        yaml.safe_dump(yml, f, default_flow_style=False)

    print(f"Pinning component {yml['name']} with version {yml['version']}.")

    return yml['name'], yml['version']


# TODO: add function to check pipeline output files are correct
def validate_successful_run():
    """Validate successful run."""
    pass


def create_copy(source: str, destination: str = None) -> "tuple[tempfile.TemporaryDirectory, str]":
    """Copy the source folder to the destination."""
    try:
        # source is a folder
        shutil.copytree(source, destination, dirs_exist_ok=True)
    except OSError as exc:  # python >2.5
        # source is a file
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(source, destination)
        else:
            raise


def set_component(component_name: str, component_version: str, component_config: dict, job_name: str) -> None:
    """Set the azureml asset name and version for the component being tested."""
    """This fills the placeholder defined at top of this file."""
    component_config["jobs"][job_name]["component"] = f"azureml:{component_name}:{component_version}"
