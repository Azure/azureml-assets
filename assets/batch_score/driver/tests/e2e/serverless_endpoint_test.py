# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

# Common configuration
source_dir = os.getcwd()
gated_pipeline_filepath = os.path.join(source_dir, "driver", "tests", "e2e", "prs_pipeline_templates", "base.yml")

RUN_NAME = "batch_score_aoai_endpoint_test"
JOB_NAME = "gated_batch_score"  # Should be equivalent to base.yml's job name
YAML_COMPONENT = {"jobs": {JOB_NAME: {"component": None}}}  # Placeholder for component name set below.
YAML_APPLICATION_INSIGHTS = {"jobs": {JOB_NAME: {
    "inputs": {
        "app_insights_connection_string": "InstrumentationKey="
    }
}}}
YAML_ENV_VARS_REDACT_PROMPTS = {"jobs": {JOB_NAME: {
    "environment_variables": {
        "BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG": "false",
    }
}}}
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": {JOB_NAME: {
    "inputs": {
        "tally_failed_requests": True
    },
    "error_threshold": 0,
    "mini_batch_error_threshold": 0
}}}

# Scoring configuration
YAML_SERVERLESS_COMPLETION_ENDPOINT = {"jobs": {JOB_NAME: {
    "inputs": {
        "api_type": "completion",
        "scoring_url": "https://llama-completion.eastus2.inference.ai.azure.com/v1/completions",
        "authentication_type": "azureml_workspace_connection",
        "connection_name": "LlamaMaasConnection"
    }
}}}

# Input data assets
YAML_COMPLETION_TEST_DATA_ASSET = {"inputs": {"pipeline_job_data_path": {"path": "azureml:e2e_llama_completion_data:1"}}}


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_serverless_endpoint_batch_score_completion(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_COMPLETION_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_SERVERLESS_COMPLETION_ENDPOINT,
                              YAML_ENV_VARS_REDACT_PROMPTS,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])
