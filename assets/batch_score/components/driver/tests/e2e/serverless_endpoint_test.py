# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains end-to-end tests for serverless endpoints."""

import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

# Common configuration
source_dir = os.getcwd()
gated_llm_pipeline_filepath = os.path.join(
    pytest.source_dir, "tests", "e2e", "prs_pipeline_templates", "base_llm.yml")

RUN_NAME = "batch_score_aoai_endpoint_test"
JOB_NAME = "gated_batch_score_llm"  # Should be equivalent to base_llm.yml's job name
YAML_COMPONENT = {"jobs": {JOB_NAME: {"component": None}}}  # Placeholder for component name set below.
YAML_ENV_VARS = {"jobs": {JOB_NAME: {
    "environment_variables": {
        "BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG": "false",
        "BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS": "True"
    }
}}}
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": {JOB_NAME: {
    "inputs": {
        # TODO: add tally_failed_requests to the file config
        # "tally_failed_requests": True
    },
    "error_threshold": 0,
    "mini_batch_error_threshold": 0
}}}

# Scoring configuration
YAML_SERVERLESS_COMPLETION_FILE_CONFIG = {
    "jobs": {
        JOB_NAME: {
            "inputs": {
                "configuration_file": {
                    "path": "azureml:serverless_completion_configuration:5",
                    "type": "uri_file",
                }
            }
        }
    }
}

# Input data assets
YAML_COMPLETION_TEST_DATA_ASSET = {"inputs": {
    "pipeline_job_data_path": {"path": "azureml:e2e_llama_completion_data:1"}}
}


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(20 * 60)
def test_gated_serverless_endpoint_batch_score_completion(llm_batch_score_yml_component):
    """Test gate for batch score serverless endpoints completion models."""
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}

    yaml_update = deep_update(
        YAML_COMPONENT,
        YAML_COMPLETION_TEST_DATA_ASSET,
        YAML_SERVERLESS_COMPLETION_FILE_CONFIG,
        YAML_ENV_VARS,
        YAML_DISALLOW_FAILED_REQUESTS,
        display_name)

    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_llm_pipeline_filepath,
        yaml_overrides=[yaml_update])
