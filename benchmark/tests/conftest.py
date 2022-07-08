import os
import sys
import logging
import pytest
import tempfile
from unittest.mock import Mock

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "components")
)

if SRC_ROOT not in sys.path:
    print(f"Adding {SRC_ROOT} to path")
    sys.path.append(str(SRC_ROOT))


@pytest.fixture()
def temporary_dir():
    """Creates a temporary directory for the tests"""
    temp_directory = tempfile.TemporaryDirectory()
    yield temp_directory.name
    temp_directory.cleanup()


@pytest.fixture()
def returned_job_mock():
    """Mock of a job returned from ml_client.jobs.create_or_update()"""
    # example of a job returned from ml_client.jobs.create_or_update()
    _returned_job_mock = Mock()
    _returned_job_mock.name = "THIS_IS_A_MOCK_NAME"

    # mock: returned_job.services["Studio"].endpoint
    _returned_job_mock.services = {"Studio": Mock()}
    _returned_job_mock.services["Studio"].endpoint = "THIS_IS_A_MOCK_URL"

    return _returned_job_mock


@pytest.fixture()
def ml_client_instance_mock(returned_job_mock):
    """Mock of an instance of a MLClient"""
    _ml_client_instance_mock = Mock()

    # example of a job returned from ml_client.jobs.create_or_update()
    _ml_client_instance_mock.jobs.create_or_update.return_value = returned_job_mock

    # mock the call to wait for completion ml_client.jobs.stream(returned_job.name)
    _ml_client_instance_mock.jobs.stream.return_value = True

    # mock the call to get dataset ml_client.datasets.get()
    _ml_client_instance_mock.datasets.get.return_value = None

    return _ml_client_instance_mock
