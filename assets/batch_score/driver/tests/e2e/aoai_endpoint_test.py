import os

import pytest
from pydantic.utils import deep_update

from .util import _submit_job_and_monitor_till_completion, set_component

# Common configuration
cpu_compute_target = "cpu-cluster"
source_dir = os.getcwd()
gated_pipeline_filepath =  os.path.join(source_dir, "driver", "tests", "e2e", "prs_pipeline_templates", "base.yml")

RUN_NAME = "batch_score_aoai_endpoint_test"
JOB_NAME = "gated_batch_score" # Should be equivalent to base.yml's job name
YAML_COMPONENT = {"jobs": { JOB_NAME: { "component": None }}} # Placeholder for component name set below.
YAML_APPLICATION_INSIGHTS = { "jobs": { JOB_NAME: {
    "inputs": {
        # https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/c0afea91-faba-4d71-bcb6-b08134f69982/resourceGroups/batchscore-test-centralus/providers/Microsoft.Insights/components/wsbatchscorece2707952777/overview
        "app_insights_connection_string": "InstrumentationKey=b8c396eb-7709-40a1-8363-527341518ab4;IngestionEndpoint=https://centralus-0.in.applicationinsights.azure.com/;LiveEndpoint=https://centralus.livediagnostics.monitor.azure.com/" 
    }
}}}
YAML_DISALLOW_FAILED_REQUESTS = {"jobs": { JOB_NAME: {
    "inputs": {
        "tally_failed_requests": True
    },
    "error_threshold": 0,
    "mini_batch_error_threshold": 0
}}}

# Scoring configuration
YAML_AOAI_COMPLETION_ENDPOINT = {"jobs": { JOB_NAME: {
    "inputs": {
        "api_type": "completion",
        "scoring_url": "https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/completions?api-version=2023-03-15-preview",
        "authentication_type": "azureml_workspace_connection",
        "connection_name": "batchscore-connection"
    }
}}}

YAML_AOAI_CHAT_COMPLETION_ENDPOINT = {"jobs": { JOB_NAME: {
    "inputs": {
        "api_type": "chat_completion",
        "scoring_url": "https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview",
        "authentication_type": "azureml_workspace_connection",
        "connection_name": "batchscore-connection"
    }
}}}

YAML_AOAI_EMBEDDINGS_ENDPOINT = {"jobs": { JOB_NAME: {
    "inputs": {
        "api_type": "embeddings",
        "scoring_url": "https://sunjoli-aoai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01",
        "authentication_type": "azureml_workspace_connection",
        "connection_name": "batchscore-connection"
    }
}}}

YAML_AOAI_COMPLETION_ENDPOINT_CONFIG_FILE = {"jobs": { JOB_NAME: {
    "inputs": {
        "api_type": "completion",
        "scoring_url": "unused",
        "configuration_file": {
            "path": "azureml://locations/centralus/workspaces/537a35e5-5a1e-4f35-9589-de93638b2e84/data/e2e_completion_scoring_config/versions/2",
            "type": "uri_file"
        }
    }
}}}

# Input data assets
YAML_AOAI_COMPLETION_TEST_DATA_ASSET = { "inputs": { "pipeline_job_data_path": { "path": "azureml:e2e_aoai_test_data:1" }}}
YAML_AOAI_CHAT_COMPLETION_TEST_DATA_ASSET = { "inputs": { "pipeline_job_data_path": { "path": "azureml:e2e_aoai_chat_completion_test_data:1" }}}
YAML_AOAI_EMBEDDING_TEST_DATA_ASSET = { "inputs": { "pipeline_job_data_path": { "path": "azureml:e2e_aoai_embedding_test_data:1" }}}

@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_completion(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_COMPLETION_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_AOAI_COMPLETION_ENDPOINT,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])
    
@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_chat_completion(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_CHAT_COMPLETION_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_AOAI_CHAT_COMPLETION_ENDPOINT,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])
    
@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_embedding(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_EMBEDDING_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_AOAI_EMBEDDINGS_ENDPOINT,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])
    
@pytest.mark.smoke
@pytest.mark.e2e
@pytest.mark.timeout(15 * 60)
def test_gated_aoai_batch_score_completion_with_scoring_config_file(llm_batch_score_yml_component):
    set_component(*llm_batch_score_yml_component, component_config=YAML_COMPONENT, job_name=JOB_NAME)
    display_name = {"display_name": f"{RUN_NAME}_smoke"}
    yaml_update = deep_update(YAML_COMPONENT,
                              YAML_AOAI_COMPLETION_TEST_DATA_ASSET,
                              YAML_APPLICATION_INSIGHTS,
                              YAML_AOAI_COMPLETION_ENDPOINT_CONFIG_FILE,
                              YAML_DISALLOW_FAILED_REQUESTS,
                              display_name)
    _submit_job_and_monitor_till_completion(
        ml_client=pytest.ml_client,
        pipeline_filepath=gated_pipeline_filepath,
        yaml_overrides=[yaml_update])