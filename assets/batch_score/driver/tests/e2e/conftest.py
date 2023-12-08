# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end test configuration."""

import os

import pytest
from azure.ai.ml import MLClient, load_component
from azure.identity import DefaultAzureCredential

from .util import _set_and_get_component_name_ver, create_copy


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
                                                                "batch_score_devops_copy.yml")


def pytest_unconfigure():
    """Unconfigure pytest."""
    print("Pytest unconfigure started.")

    # Delete copied component to not muddle dev environments
    try:
        os.remove(pytest.copied_batch_score_component_filepath)
    except FileNotFoundError:
        pass


@pytest.fixture(scope="module")
def batch_score_yml_component():
    """Register batch score component."""
    return register_component("batch_score.yml")


@pytest.fixture(scope="module")
def batch_score_embeddings_yml_component():
    """Register batch score embeddings component."""
    return register_component("batch_score_embeddings.yml")


@pytest.fixture(scope="module")
def batch_score_vesta_chat_completion_yml_component():
    """Register batch score vesta chat completion component."""
    return register_component("batch_score_vesta_chat_completion.yml")


@pytest.fixture(scope="module")
def llm_batch_score_yml_component():
    """Register llm batch score component."""
    return register_component("batch_score_llm.yml")


def register_component(component_yml_name):
    """Register component."""
    # Copy component to a temporary file to not muddle dev environments
    batch_score_component_filepath = os.path.join(pytest.source_dir, "yamls", "components", "v2", component_yml_name)
    create_copy(batch_score_component_filepath, pytest.copied_batch_score_component_filepath)

    # pins batch_component version
    component_name, component_version = _set_and_get_component_name_ver(
        pytest.copied_batch_score_component_filepath)

    # registers the specified component from local yaml
    batch_component = load_component(source=pytest.copied_batch_score_component_filepath)
    batch_component = pytest.ml_client.create_or_update(batch_component)
    print(f"Component {component_name} with version {component_version} is registered")
    return component_name, component_version
