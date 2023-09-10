# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Global test level fixtures."""

import os
import shutil
import uuid
import pytest
from tests.mocks.mock_runmetric_client import MockRunMetricClient


@pytest.fixture(scope="session")
def root_temporary_directory():
    """Return the path to the root temporary directory."""
    directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".temp")
    os.makedirs(directory, exist_ok=True)
    yield directory

    # Don't remove the temp directory if running in Azure Pipelines
    if "TF_BUILD" in os.environ:
        return
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="function")
def unique_temporary_directory(root_temporary_directory):
    """Return the path to a unique temporary directory used to capture test's resource output."""
    directory = os.path.join(root_temporary_directory, str(uuid.uuid4()))
    os.makedirs(directory, exist_ok=True)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_runmetric_client():
    """Mock run metrics client which returns a static GUID as run ID and no-ops on  publishing."""
    return MockRunMetricClient()


@pytest.fixture(scope="function")
def signal_name():
    """Mock signal name."""
    return "signal_name"


@pytest.fixture(scope="function")
def monitor_name():
    """Mock monitor name."""
    return "monitor_name"
