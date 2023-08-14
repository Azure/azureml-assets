# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures for model monitoring component tests."""

from datetime import datetime
import os
from glob import glob
from typing import List
import time

from azure.ai.ml import MLClient, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Job, Data
from azure.identity import AzureCliCredential
import pytest

from tests.e2e.utils.io_utils import (
    write_to_yaml,
    load_from_yaml,
    print_header,
    generate_random_filename,
)

lock_file = ".lock"


def _get_subscription_id():
    return os.environ.get("SUBSCRIPTION_ID", "ea4faa5b-5e44-4236-91f6-5483d5b17d14")


def _get_tenant_id():
    return os.environ.get("TENANT_ID", "72f988bf-86f1-41af-91ab-2d7cd011db47")


def _get_resource_group():
    return os.environ.get("RESOURCE_GROUP", "model-monitoring-canary-rg")


def _get_workspace_name():
    return os.environ.get("WORKSPACE_NAME", "model-monitoring-canary-ws")


def _is_main_worker(worker_id):
    return worker_id == "gw0" or worker_id == "master"


def _watch_file(file: str, timeout_in_seconds):
    seconds_elapsed = 0
    while not os.path.exists(file):
        time.sleep(1)
        seconds_elapsed += 1
        if seconds_elapsed >= timeout_in_seconds:
            break


def _create_data_asset(ml_client: MLClient, name: str, path: str, type: AssetTypes):
    my_data = Data(
        path=path,
        type=type,
        name=name,
        version='1'
    )
    print(f"Creating a {type} data asset with name '{name}'")
    ml_client.data.create_or_update(my_data)


@pytest.fixture(scope="session")
def main_worker_lock(worker_id):
    """Lock until the main worker releases its lock."""
    if _is_main_worker(worker_id):
        return worker_id

    _watch_file(file=lock_file, timeout_in_seconds=120)
    return worker_id


@pytest.fixture(scope="session")
def ml_client() -> MLClient:
    """Return a MLClient used to manage AML resources."""
    ws = MLClient(
        AzureCliCredential(tenant_id=_get_tenant_id()),
        subscription_id=_get_subscription_id(),
        resource_group_name=_get_resource_group(),
        workspace_name=_get_workspace_name(),
    )
    return ws


@pytest.fixture(scope="session")
def e2e_resources_directory(components_directory):
    """Return the path to the directory holding model monitoring's e2e resources."""
    return os.path.abspath(os.path.join(components_directory, "tests", "e2e", "resources"))


@pytest.fixture(scope="session", autouse=True)
def register_data_assets(main_worker_lock, ml_client, e2e_resources_directory) -> MLClient:
    """Return a MLClient used to manage AML resources."""
    if not _is_main_worker(main_worker_lock):
        return

    registered_data_assets = [x.name for x in ml_client.data.list()]
    for directory in glob(f"{e2e_resources_directory}/*", recursive=False):
        name = os.path.basename(directory)

        if name not in registered_data_assets:
            if "mltable" in name:
                _create_data_asset(ml_client, name, directory, AssetTypes.MLTABLE)
            if "uri_folder" in name:
                _create_data_asset(ml_client, name, directory, AssetTypes.URI_FOLDER)


@pytest.fixture(scope="session")
def asset_version(main_worker_lock):
    """Return the asset version for this run."""
    # Ensure all workers leverages the same asset versions
    # Main worker is gw0 - everyone else will be reading the version the main worker has created.
    version_file = ".version"
    if main_worker_lock == "gw0" or main_worker_lock == "master":
        version = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        with open(version_file, "w") as fp:
            fp.write(version)
        yield version
        os.remove(version_file)
        return

    _watch_file(file=version_file, timeout_in_seconds=120)
    version = ""
    with open(version_file, "r") as fp:
        version = fp.read()
    yield version


@pytest.fixture(scope="session")
def model_monitoring_root_directory() -> str:
    """Return the path to model monitoring's root directory."""
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
    )


@pytest.fixture(scope="session")
def components_directory(model_monitoring_root_directory) -> str:
    """Return the path to the directory holding model monitoring's components."""
    return os.path.abspath(os.path.join(model_monitoring_root_directory, "components"))


@pytest.fixture(scope="session")
def source_directory(components_directory) -> str:
    """Return the path to the directory holding test resources."""
    return os.path.abspath(os.path.join(components_directory, "src"))


@pytest.fixture(scope="session", autouse=True)
def model_monitoring_component_specs(components_directory) -> List[dict]:
    """Return the dictionary representation of all model monitoring component yaml definitions."""
    specs = []
    import glob

    for file in glob.glob(
        os.path.join(components_directory, "**/spec.yaml"), recursive=True
    ):
        specs.append(file)
    return specs


@pytest.fixture(scope="session", autouse=True)
def model_monitoring_components(model_monitoring_component_specs) -> List[dict]:
    """Return the dictionary representation of all model monitoring component yaml definitions."""
    components = []
    for file in model_monitoring_component_specs:
        components.append(load_from_yaml(file))
    return components


@pytest.fixture(scope="session", autouse=True)
def test_suite_name() -> str:
    """Name of the test suite."""
    return os.environ.get("GITHUB_REF_NAME", "local").replace("/", "_")


@pytest.fixture(scope="session", autouse=True)
def publish_command_components(
    main_worker_lock,
    model_monitoring_components,
    root_temporary_directory,
    asset_version,
    ml_client: MLClient,
    source_directory,
):
    """Publish all of the command components used by data drift to the test workspace."""
    if not _is_main_worker(main_worker_lock):
        return

    print_header("Publishing Data Drift Command Components")
    out_directory = os.path.join(root_temporary_directory, "command_components")
    os.makedirs(out_directory, exist_ok=True)

    for component in model_monitoring_components:
        if component["type"] != "spark":
            continue
        print(f"Publishing {component['name']}:{component['version']}")
        component["code"] = source_directory
        component["version"] = asset_version
        spec_path = os.path.join(
            out_directory, f"{component['name']}-{generate_random_filename('yaml')}"
        )
        write_to_yaml(spec_path, component)

        ml_client.components.create_or_update(load_component(spec_path))

        print(f"Successfully published {component['name']}.")
        

@pytest.fixture(scope="session", autouse=True)
def publish_mm_az_mon_proto_component(
    main_worker_lock,
    publish_command_components,
    model_monitoring_components,
    components_directory,
    root_temporary_directory,
    asset_version,
    ml_client,
):
    """Publish the data drift model monitor pipeline component to the test workspace."""
    if not _is_main_worker(main_worker_lock):
        return

    print_header("Publishing mm az mon proto")
    out_directory = os.path.join(root_temporary_directory, "command_components")
    os.makedirs(out_directory, exist_ok=True)

    for component in model_monitoring_components:
        if component["name"] != "mm_az_mon_proto_wrapper":
            continue
        print(f"Publishing {component['name']}..")
        component["jobs"]["mm_az_mon_proto_spark"][
            "component"
        ] = f"azureml:mm_az_mon_proto_spark:{asset_version}"
        component["version"] = asset_version

        spec_path = os.path.join(
            out_directory, f"{component['name']}-{generate_random_filename('yaml')}"
        )

        write_to_yaml(spec_path, component)
        ml_client.components.create_or_update(load_component(spec_path))
        print(f"Successfully published {component['name']}.")


@pytest.fixture(scope="session", autouse=True)
def release_lock(
    # publish_data_quality_model_monitor_component,
    # publish_prediction_drift_model_monitor_component,
    # publish_data_drift_model_monitor_component,
    # publish_feature_attr_drift_signal_monitor_component,
    publish_command_components,
    publish_mm_az_mon_proto_component,
    main_worker_lock,
):
    """Release the main worker lock."""
    if _is_main_worker(main_worker_lock):
        with open(lock_file, "w"):
            pass
        yield
        os.remove(lock_file)
    else:
        yield


@pytest.fixture(scope="session", autouse=True)
def get_component(ml_client, asset_version):
    """Get a component from the test workspace with a given name."""

    def _get(component_name):
        return ml_client.components.get(name=component_name, version=asset_version)

    return _get


@pytest.fixture(scope="function", autouse=True)
def download_job_output(ml_client, unique_temporary_directory):
    """Download a named job output."""

    def download_output(job: Job, output_name: str) -> str:
        ml_client.jobs.download(
            name=job.name,
            download_path=unique_temporary_directory,
            output_name=output_name,
        )
        return os.path.join(unique_temporary_directory, "named-outputs", output_name)

    return download_output
