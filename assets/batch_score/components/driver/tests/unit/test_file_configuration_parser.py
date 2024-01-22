import os
from pathlib import Path

import pytest

from src.batch_score.common.configuration.file_configuration_parser import (
    FileConfigurationParser,
)
from src.batch_score.common.configuration.file_configuration_validator import (
    FileConfigurationValidator,
)


_current_file_path = Path(os.path.abspath(__file__))
configs_root = (
    _current_file_path.parent.parent.parent
    / "test_assets"
    / "configuration_files"
    / "for_e2e_tests"
)

@pytest.mark.parametrize(
    "file_name, override_expected_config",
    [
        (
            "aoai_completion.json",
            {
                'api_type': 'completion',
                'connection_name': 'batchscore-connection',
                'scoring_url': 'https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/completions?api-version=2023-03-15-preview',
                'segment_large_requests': True,
                'segment_max_token_size': 1000,
            },
        ),
        (
            "aoai_chat_completion.json",
            {
                'api_type': 'chat_completion',
                'connection_name': 'batchscore-connection',
                'scoring_url': 'https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview',
            },
        ),
        (
            "aoai_embedding.json",
            {
                'api_type': 'embedding',
                'batch_size_per_request': 2,
                'connection_name': 'batchscore-connection',
                'scoring_url': 'https://sunjoli-aoai.openai.azure.com/openai/deployments/text-embedding-ada-002/embeddings?api-version=2022-12-01',
            },
        ),
        (
            "serverless_completion.json",
            {
                'api_type': 'completion',
                'connection_name': 'LlamaMaasConnection',
                'scoring_url': 'https://llama-completion.eastus2.inference.ai.azure.com/v1/completions',
                'segment_large_requests': True,
                'segment_max_token_size': 1000,
            },
        ),
    ],
)
def test_aoai_completion(file_name, override_expected_config):
    args = [
        "--async_mode", "True",
        "--configuration_file", str(configs_root / file_name),
        "--partitioned_scoring_results", "~/resources/mini_batch_results_output_directory",
    ]
    parser = FileConfigurationParser(FileConfigurationValidator())
    configuration = parser.parse_configuration(args)

    expected_configuration = get_base_configuration()
    for key, value in override_expected_config.items():
        expected_configuration[key] = value

    assert vars(configuration) == expected_configuration

def get_base_configuration():
    return {
        "additional_headers": "{}",
        "additional_properties": "{}",
        "api_key_name": None,
        "api_type": "completion",
        "app_insights_connection_string": None,
        "app_insights_log_level": "debug",
        "async_mode": False,
        "authentication_type": "connection",
        "batch_pool": None,
        "batch_size_per_request": 1,
        "configuration_file": None,
        "configuration_file": None,
        "connection_name": "batchscore-connection",
        "debug_mode": None,
        "ensure_ascii": False,
        "image_input_folder": None,
        "initial_worker_count": 100,
        "max_retry_time_interval": 600,
        "max_worker_count": 200,
        "mini_batch_results_out_directory": '~/resources/mini_batch_results_output_directory',
        "online_endpoint_url": None,
        "output_behavior": "summary_only",
        "quota_audience": None,
        "quota_estimator": None,
        "request_path": None,
        "save_mini_batch_results": "enabled",
        "scoring_url": "https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/completions?api-version=2023-03-15-preview",
        "segment_large_requests": False,
        "segment_max_token_size": 0,
        "service_namespace": None,
        "stdout_log_level": "debug",
        "tally_exclusions": None,
        "tally_failed_requests": None,
        "token_file_path": None,
        "user_agent_segment": None,
    }