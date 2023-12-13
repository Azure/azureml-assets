# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains end-to-end tests for AOAI endpoints."""

import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

# Common configuration
cpu_compute_target = "cpu-cluster"
source_dir = os.getcwd()
gated_llm_pipeline_filepath = os.path.join(
    source_dir, "driver", "tests", "e2e", "prs_pipeline_templates", "base_llm.yml")

RUN_NAME = "batch_score_aoai_endpoint_test"
JOB_NAME = "gated_batch_score_llm"  # Should be equivalent to base_llm.yml's job name
YAML_COMPONENT = {"jobs": {JOB_NAME: {"component": None}}}  # Placeholder for component name set below.
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": {JOB_NAME: {
    "inputs": {
        # TODO: add tally_failed_requests to the file config
        # "tally_failed_requests": True
    },
    "error_threshold": 0,
    "mini_batch_error_threshold": 0
}}}


# Scoring configuration
def _get_file_config_yaml(data_asset_path: str):
    return {
        "jobs": {
            JOB_NAME: {
                "inputs": {
                    "configuration_file": {
                        "path": data_asset_path,
                        "type": "uri_file",
                    }
                }
            }
        }
    }


YAML_AOAI_COMPLETION_FILE_CONFIG = _get_file_config_yaml("azureml:aoai_completion_configuration:5")
YAML_AOAI_CHAT_COMPLETION_FILE_CONFIG = _get_file_config_yaml("azureml:aoai_chat_completion_configuration:5")
YAML_AOAI_EMBEDDING_FILE_CONFIG = _get_file_config_yaml("azureml:aoai_embedding_configuration:5")

# Input data assets
YAML_AOAI_COMPLETION_TEST_DATA_ASSET = {"inputs": {
    "pipeline_job_data_path": {
        "path": "azureml:e2e_aoai_test_data:1"
    }
}}
YAML_AOAI_CHAT_COMPLETION_TEST_DATA_ASSET = {"inputs": {
    "pipeline_job_data_path": {
        "path": "azureml:e2e_aoai_chat_completion_test_data:1"
    }
}}
YAML_AOAI_EMBEDDING_TEST_DATA_ASSET = {"inputs": {
    "pipeline_job_data_path": {
        "path": "azureml:e2e_aoai_embedding_test_data:1"
    }
}}


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_completion(llm_batch_score_yml_component):
    """Test gate for AOAI batch score completion model."""
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_COMPLETION_TEST_DATA_ASSET,
                              YAML_AOAI_COMPLETION_FILE_CONFIG,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_llm_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_chat_completion(llm_batch_score_yml_component):
    """Test gate for AOAI batch score chat completion model."""
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_CHAT_COMPLETION_TEST_DATA_ASSET,
                              YAML_AOAI_CHAT_COMPLETION_FILE_CONFIG,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_llm_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_embedding(llm_batch_score_yml_component):
    """Test gate for AOAI batch score embedding model."""
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_EMBEDDING_TEST_DATA_ASSET,
                              YAML_AOAI_EMBEDDING_FILE_CONFIG,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_llm_pipeline_filepath,
        yaml_overrides=[yaml_update])
