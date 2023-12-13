# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end test configuration."""

from datetime import datetime
import time
import os
import uuid

import pytest
from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

from .util import _get_component_name, _set_and_get_component_name_ver, create_copy


lock_file = ".lock"


def _is_main_worker(worker_id):
    return worker_id == "gw0" or worker_id == "master"


def _watch_file(file: str, timeout_in_seconds):
    seconds_elapsed = 0
    while not os.path.exists(file):
        time.sleep(1)
        seconds_elapsed += 1
        if seconds_elapsed >= timeout_in_seconds:
            break


@pytest.fixture(scope="session")
def main_worker_lock(worker_id):
    """Lock until the main worker releases its lock."""
    if _is_main_worker(worker_id):
        return worker_id
    _watch_file(file=lock_file, timeout_in_seconds=120)
    return worker_id


@pytest.fixture(scope="session", autouse=True)
def release_lock(main_worker_lock, register_components):
    """Release the main worker lock."""
    if _is_main_worker(main_worker_lock):
        with open(lock_file, "w"):
            pass
        yield
        os.remove(lock_file)
    else:
        yield


@pytest.fixture(scope="session")
def asset_version(main_worker_lock):
    """Return the asset version for this run."""
    # Ensure all workers leverages the same asset versions
    # Main worker is gw0 - everyone else will be reading the version the main worker has created.

    version_file = ".version"
    if _is_main_worker(main_worker_lock):
        version = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        with open(version_file, "w") as fp:
            fp.write(version)

        yield version
        os.remove(version_file)
    else:
        _watch_file(file=version_file, timeout_in_seconds=120)
        version = ""
        with open(version_file, "r") as fp:
            version = fp.read()
        yield version


@pytest.fixture(autouse=True, params=[pytest.param(None, marks=pytest.mark.e2e)])
def mark_as_e2e_test():
    """Mark all tests in this directory as unit tests."""
    pass


def pytest_configure():
    """Configure pytest."""
    print("Pytest configure started.")
    # ML_Client set up
    pytest.ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id="c0afea91-faba-4d71-bcb6-b08134f69982",
        resource_group_name="batchscore-test-centralus",
        workspace_name="ws-batchscore-centralus",
    )

    # Prepare to copy components in fixtures below to a temporary file to not muddle dev environments
    pytest.source_dir = os.getcwd()
    pytest.copied_batch_score_component_filepath = os.path.join(pytest.source_dir,
                                                                "yamls",
                                                                "components",
                                                                "v2",
                                                                f"{str(uuid.uuid4())}_batch_score_devops_copy.yml")


def pytest_unconfigure():
    """Unconfigure pytest."""
    print("Pytest unconfigure started.")

    # Delete copied component to not muddle dev environments
    try:
        os.remove(pytest.copied_batch_score_component_filepath)
    except FileNotFoundError:
        pass


@pytest.fixture(autouse=True, scope="session")
def register_components(main_worker_lock, asset_version):
    """Register components."""
    if not _is_main_worker(main_worker_lock):
        return

    register_component("batch_score.yml", asset_version)
    register_component("batch_score_embeddings.yml", asset_version)
    register_component("../llm/batch_score_llm.yml", asset_version)
    register_component("batch_score_vesta_chat_completion.yml", asset_version)


@pytest.fixture(scope="session")
def batch_score_yml_component(asset_version):
    """Get batch score component."""
    return _get_component_metadata("batch_score.yml", asset_version)


@pytest.fixture(scope="session")
def batch_score_embeddings_yml_component(asset_version):
    """Get batch score embeddings component."""
    return _get_component_metadata("batch_score_embeddings.yml", asset_version)


@pytest.fixture(scope="session")
def llm_batch_score_yml_component(asset_version):
    """Get batch score llm component."""
    return _get_component_metadata("../llm/batch_score_llm.yml", asset_version)


@pytest.fixture(scope="session")
def batch_score_vesta_chat_completion_yml_component(asset_version):
    """Get batch score vesta chat completion component."""
    return _get_component_metadata("batch_score_vesta_chat_completion.yml", asset_version)


def register_component(component_yml_name, asset_version):
    """Register component."""
    # Copy component to a temporary file to not muddle dev environments
    batch_score_component_filepath = os.path.join(pytest.source_dir, "yamls", "components", "v2", component_yml_name)
    create_copy(batch_score_component_filepath, pytest.copied_batch_score_component_filepath)

    # pins batch_component version
    component_name, component_version = _set_and_get_component_name_ver(
        pytest.copied_batch_score_component_filepath, asset_version)
    print(f"Component Name: {component_name}, Version: {component_version}.")

    # registers the specified component from local yaml
    batch_component = load_component(
        source=pytest.copied_batch_score_component_filepath)
    batch_component = pytest.ml_client.create_or_update(batch_component)
    print(f"Component {component_name} with version {component_version} is registered")
    return component_name, component_version


def _get_component_metadata(component_yml_name, asset_version):
    batch_score_component_filepath = os.path.join(
        pytest.source_dir, "yamls", "components", "v2", component_yml_name
    )
    return _get_component_name(batch_score_component_filepath), asset_version
