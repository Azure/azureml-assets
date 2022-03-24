"""
Executes the series of scripts end-to-end
to test LightGBM (python) manual benchmark
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch, Mock
import argparse

from pipelines.common.main import main as common_main

# IMPORTANT: see conftest.py for fixtures

@patch('pipelines.common.main.MLClient') # patch the import
def test_common_main(azure_ml_mlclient_mock, ml_client_instance_mock):
    """We're just testing if the code works on a basic level"""
    azure_ml_mlclient_mock.return_value = ml_client_instance_mock

    _build_mock = Mock()
    _build_mock.return_value = "fake_pipeline_object"

    _build_arguments_mock = Mock()

    # create test arguments for the script
    script_args = [
        "train.py",
        "--aml_subscription_id", "FAKE_SUB",
        "--aml_resource_group_name", "FAKE_RG",
        "--aml_workspace_name", "FAKE_WS",
        "--experiment_name", "FAKE_EXP",
        "--wait_for_completion",
    ]
    # replaces sys.argv with test arguments and run main
    with patch.object(sys, "argv", script_args):
        common_main(_build_mock, _build_arguments_mock)

    # call to MLClient
    mlclient_call_args, mlclient_call_kwargs = azure_ml_mlclient_mock.call_args
    assert "subscription_id" in mlclient_call_kwargs and mlclient_call_kwargs["subscription_id"] == "FAKE_SUB"
    assert "resource_group_name" in mlclient_call_kwargs and mlclient_call_kwargs["resource_group_name"] == "FAKE_RG"
    assert "workspace_name" in mlclient_call_kwargs and mlclient_call_kwargs["workspace_name"] == "FAKE_WS"

    # call to build + build_arguments
    _build_arguments_mock.assert_called_once()
    _build_mock.assert_called_once()

    # call to submit the job
    ml_client_instance_mock.jobs.create_or_update.assert_called_with(
        "fake_pipeline_object",
        experiment_name="FAKE_EXP",
        continue_run_on_step_failure=True,
    )

    # call to wait for completion
    ml_client_instance_mock.jobs.stream.assert_called_with(
        # see conftest.py, returned_job fixture is using this fake name
        "THIS_IS_A_MOCK_NAME"
    )
