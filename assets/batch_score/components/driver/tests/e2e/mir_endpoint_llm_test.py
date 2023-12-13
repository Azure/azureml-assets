import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

cpu_compute_target = "cpu-cluster"

source_dir = os.getcwd()
gated_llm_pipeline_filepath = os.path.join(
    source_dir, "driver", "tests", "e2e", "prs_pipeline_templates", "base_llm.yml")

RUN_NAME = "batch_devops_test"
JOB_NAME = "gated_batch_score_llm"  # Should be equivalent to base_llm.yml's job name
YAML_COMPONENT = {"jobs": {JOB_NAME: {"component": None}}}  # Placeholder for component name set below.

YAML_SMOKE_TEST_DATA_ASSET = {"inputs": {"pipeline_job_data_path": {"path": "azureml:e2e_smoke_test_data:6"}}}
YAML_SERVERLESS_COMPLETION_FILE_CONFIG = {
    "jobs": {
        JOB_NAME: {
            "inputs": {
                "configuration_file": {
                    "path": "azureml:mir_completion_configuration:5",
                    "type": "uri_file",
                }
            }
        }
    }
}
YAML_ENV_VARS_REDACT_PROMPTS = {"jobs": {JOB_NAME: {
    "environment_variables": {
        "BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG": "false",
    }
}}}
# If tally_failed_requests is True, the batch score job will fail if any requests fail.
# This is useful for testing scenarios where no failures are expected.
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": {JOB_NAME: {
    "inputs": {
        # TODO: add tally_failed_requests to the file config
        # "tally_failed_requests": True
    },
    "error_threshold": 0,
    "mini_batch_error_threshold": 0
}}}


# This test confirms that we can score an MIR endpoint
# using the scoring_url parameter and the batch_score_llm.yml component.
@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_batch_score_single_endpoint_using_soring_url_parameter(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_TEST_DATA_ASSET,
                              YAML_SERVERLESS_COMPLETION_FILE_CONFIG,
                              YAML_ENV_VARS_REDACT_PROMPTS,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_llm_pipeline_filepath,
        yaml_overrides=[yaml_update])
