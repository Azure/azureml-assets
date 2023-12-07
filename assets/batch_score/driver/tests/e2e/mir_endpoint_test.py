# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

cpu_compute_target = "cpu-cluster"

source_dir = os.getcwd()
gated_pipeline_filepath = os.path.join(source_dir, "driver", "tests", "e2e", "prs_pipeline_templates", "base.yml")

RUN_NAME = "batch_devops_test"
JOB_NAME = "gated_batch_score"  # Should be equivalent to base.yml's job name
YAML_COMPONENT = {"jobs": {JOB_NAME: {"component": None}}}  # Placeholder for component name set below.

YAML_SMOKE_TEST_DATA_ASSET = {"inputs": {"pipeline_job_data_path": {"path": "azureml:e2e_smoke_test_data:6"}}}
YAML_SMOKE_EMBEDDINGS_TEST_DATA_ASSET = {
    "inputs": {"pipeline_job_data_path": {"path": "azureml:cnndailymail500:1"}},
    "jobs": {JOB_NAME: {
        "inputs": {
            "batch_size_per_request": 5,
            "initial_worker_count": 5,
            "max_worker_count": 10},
        "mini_batch_size": "50kb"}}}
YAML_SMOKE_VESTA_CHAT_COMPLETION_TEST_DATA_ASSET = {
    "inputs": {"pipeline_job_data_path": {"path": "azureml:vestainput:12"}},
    "jobs": {JOB_NAME: {
        "inputs": {
            "initial_worker_count": 2,
            "max_worker_count": 10},
        "mini_batch_size": "50kb"}}}

YAML_NIGHTLY_TEST_DATA_ASSET = {"inputs": {"pipeline_job_data_path": {"path": "azureml:bing_tail_queries:1"}}}
YAML_LONGHAUL_TEST_DATA_ASSET = {"inputs": {"pipeline_job_data_path": {"path": "azureml:nl_big:1"}}}
YAML_APPLICATION_INSIGHTS = {"jobs": {JOB_NAME: {
    "inputs": {
        "app_insights_connection_string": "InstrumentationKey="
    }
}}}
YAML_GLOBAL_POOL = {"jobs": {JOB_NAME: {
    "inputs": {
        "batch_pool": "batch-score-dv3",
        "service_namespace": "batch-score-dv3",
        "quota_audience": "percent100"  # percent10-percent100 in increments of 10 are available
    }
}}}
YAML_SINGLE_ENDPOINT = {"jobs": {JOB_NAME: {
    "inputs": {
        "online_endpoint_url": "https://real-dv3-stable.centralus.inference.ml.azure.com/v1/engines/davinci/completions"  # Using shared dv3 endpoint
    }
}}}
YAML_SINGLE_ENDPOINT_USING_SCORING_URL_PARAMETER = {"jobs": {JOB_NAME: {
    "inputs": {
        "scoring_url": "https://real-dv3-stable.centralus.inference.ml.azure.com/v1/engines/davinci/completions"  # Using shared dv3 endpoint
    }
}}}
YAML_SINGLE_ENDPOINT_EMBEDDINGS = {"jobs": {JOB_NAME: {
    "inputs": {
        "online_endpoint_url": "https://real-ada-stable.centralus.inference.ml.azure.com/v1/engines/davinci/embeddings"  # Using shared dv3 endpoint
    }
}}}
YAML_ENV_VARS_REDACT_PROMPTS = {"jobs": {JOB_NAME: {
    "environment_variables": {
        "BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG": "false",
    }
}}}
# If tally_failed_requests is True, the batch score job will fail if any requests fail.
# This is useful for testing scenarios where no failures are expected.
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": {JOB_NAME: {
    "inputs": {
        "tally_failed_requests": True
    }
}}}

YAML_ASYNC_MODE_ENABLED = {"jobs": {JOB_NAME: {
    "inputs": {
        "async_mode": "true"
    }
}}}


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
@pytest.mark.parametrize("async_mode", [False, True])
def test_gated_batch_score(batch_score_yml_component, async_mode):
    set_component(*batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_GLOBAL_POOL,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              YAML_ASYNC_MODE_ENABLED if async_mode else {},
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_embeddings_batch_score(batch_score_embeddings_yml_component):
    set_component(*batch_score_embeddings_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    ada_pool = {"jobs": {JOB_NAME: {"inputs": {"batch_pool": "batch-score-ada"}}}}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_EMBEDDINGS_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_GLOBAL_POOL,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name,
                              ada_pool)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
@pytest.mark.skip(reason="Enable this after creating a test endpoint for vesta")
def test_gated_vesta_chat_completion_batch_score(batch_score_vesta_chat_completion_yml_component):
    set_component(*batch_score_vesta_chat_completion_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    batch_pool = {"jobs": {JOB_NAME: {"inputs": {"batch_pool": "gptv-eval-global", "quota_audience": "common", "service_namespace": "prometheus"}}}}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_VESTA_CHAT_COMPLETION_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_GLOBAL_POOL,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name,
                              batch_pool)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_batch_score_single_endpoint(batch_score_yml_component):
    set_component(*batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_SINGLE_ENDPOINT,
                              YAML_ENV_VARS_REDACT_PROMPTS,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


# This test confirms that we can score an MIR endpoint using the scoring_url parameter and the batch_score_llm.yml component.
@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_batch_score_single_endpoint_using_soring_url_parameter(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_SINGLE_ENDPOINT_USING_SCORING_URL_PARAMETER,
                              YAML_ENV_VARS_REDACT_PROMPTS,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_embeddings_batch_score_single_endpoint(batch_score_embeddings_yml_component):
    set_component(*batch_score_embeddings_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_SMOKE_EMBEDDINGS_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_SINGLE_ENDPOINT_EMBEDDINGS,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.timeout(8 * 60 * 60)
def test_nightly_batch_score(batch_score_yml_component):
    set_component(*batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_nightly"}
    yaml_2_by_2 = {"jobs": {JOB_NAME: {
        "max_concurrency_per_instance": 2,
        "resources": {
            "instance_count": 2
        }
    }}}
    mini_batch_size_300kb = {"jobs": {JOB_NAME: {
        "mini_batch_size": "300kb"
    }}}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_NIGHTLY_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_GLOBAL_POOL,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              yaml_2_by_2,
                              display_name,
                              mini_batch_size_300kb)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])


@pytest.mark.longhaul
@pytest.mark.e2e
@pytest.mark.timeout(2 * 24 * 60 * 60)
def test_longhaul_batch_score(batch_score_yml_component):
    set_component(*batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_longhaul"}
    yaml_2_by_2 = {"jobs": {JOB_NAME: {
        "max_concurrency_per_instance": 2,
        "resources": {
            "instance_count": 2
        }
    }}}
    mini_batch_size_300kb = {"jobs": {JOB_NAME: {
        "mini_batch_size": "300kb"
    }}}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_LONGHAUL_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_GLOBAL_POOL,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              yaml_2_by_2,
                              display_name,
                              mini_batch_size_300kb)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])
