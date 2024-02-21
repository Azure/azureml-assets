# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import json
import os
from typing import Optional, Dict
import pytest
import sys
import tempfile

from ..test_utils import get_src_dir, get_current_path
from aml_benchmark.utils.constants import AuthenticationType


sys.path.append(get_src_dir())
print(get_src_dir())

from aml_benchmark.batch_config_generator.main import main as batch_config_generator  # noqa: E402


class TestConfigGenerator:
    """Test config generation for batch score component."""

    @pytest.mark.parametrize('scoring_url, authentication_type', [
        ("https://sample.sample_region.inference.ml.azure.com/score",
         AuthenticationType.AZUREML_WORKSPACE_CONNECTION),
        ("https://demo.api.cognitive.microsoft.com/openai/deployments/demo/chat/completions?api-version=2023-07-01-preview",
         AuthenticationType.MANAGED_IDENTITY)
    ])
    @pytest.mark.parametrize('debug_mode, ensure_ascii, additional_headers, deployment_name, max_retry_time_interval, app_insights_connection_string, override_config_file_path, connection_name, expected_headers', [
        (True, False, None, "sample-deployment",
         None, '', None, 'sample_connection_name', {'azureml-model-deployment': 'sample-deployment'}),
        (False, True, '{"header-key": "header-value"}', None,
         0, 'some-appinsights-connection-string', None, None, {'header-key': 'header-value'}),
        (False, True, '{"header-key": "header-value"}', None, 20, None,
         'resource_manager_output_config.json', 'sample_connection_name', {'header-key': 'header-value'}),
    ])
    @pytest.mark.parametrize('initial_worker_count', [7])
    @pytest.mark.parametrize('max_worker_count', [10])
    @pytest.mark.parametrize('response_segment_size', [0])
    def test_e2e_config_generator(
        self,
        scoring_url: str,
        connection_name: Optional[str],
        authentication_type: AuthenticationType,
        debug_mode: bool,
        ensure_ascii: bool,
        initial_worker_count: int,
        max_worker_count: int,
        additional_headers: Optional[str],
        deployment_name: Optional[str],
        max_retry_time_interval: Optional[int],
        response_segment_size: Optional[int],
        app_insights_connection_string: Optional[str],
        override_config_file_path: Optional[str],
        expected_headers: Dict[str, str],
    ):
        """Test config generation."""
        with tempfile.TemporaryDirectory() as d:
            config_file_path = os.path.join(d, "config.json")
            if override_config_file_path:
                override_config_file_path =os.path.join(get_current_path(), "data", override_config_file_path)
            if not connection_name and authentication_type is AuthenticationType.AZUREML_WORKSPACE_CONNECTION:
                with pytest.raises(Exception):
                    batch_config_generator(
                        configuration_file=override_config_file_path,
                        scoring_url=scoring_url,
                        connection_name=connection_name,
                        authentication_type=authentication_type,
                        debug_mode=debug_mode,
                        ensure_ascii=ensure_ascii,
                        initial_worker_count=initial_worker_count,
                        max_worker_count=max_worker_count,
                        batch_score_config_path=config_file_path,
                        additional_headers=additional_headers,
                        deployment_name=deployment_name,
                        max_retry_time_interval=max_retry_time_interval,
                        response_segment_size=response_segment_size,
                        app_insights_connection_string=app_insights_connection_string,
                    )
            else:
                batch_config_generator(
                    configuration_file=override_config_file_path,
                    scoring_url=scoring_url,
                    connection_name=connection_name,
                    authentication_type=authentication_type,
                    debug_mode=debug_mode,
                    ensure_ascii=ensure_ascii,
                    initial_worker_count=initial_worker_count,
                    max_worker_count=max_worker_count,
                    batch_score_config_path=config_file_path,
                    additional_headers=additional_headers,
                    deployment_name=deployment_name,
                    max_retry_time_interval=max_retry_time_interval,
                    response_segment_size=response_segment_size,
                    app_insights_connection_string=app_insights_connection_string,
                )
                config_output = json.load(open(config_file_path, 'r'))
                override_config = json.load(open(override_config_file_path, 'r')) if override_config_file_path else {}
                expected_scoring_url = override_config.get('scoring_url', scoring_url)
                expected_connection_name = override_config.get('connection_name', connection_name)
                assert config_output['api']['type'] == 'completion'
                assert config_output['authentication']['type'] == ('connection' if authentication_type is AuthenticationType.AZUREML_WORKSPACE_CONNECTION else 'managed_identity')
                if expected_connection_name and authentication_type is AuthenticationType.AZUREML_WORKSPACE_CONNECTION:
                    assert config_output['authentication']['name'] == expected_connection_name
                else:
                    assert 'name' not in config_output['authentication']
                if "inference.ml" in expected_scoring_url:
                    assert config_output['inference_endpoint']['type'] == 'azureml_online_endpoint'
                elif "inference.ai" in expected_scoring_url:
                    assert config_output['inference_endpoint']['type'] == 'azureml_serverless_endpoint'
                elif 'openai' in expected_scoring_url:
                    assert config_output['inference_endpoint']['type'] == 'azure_openai'
                    if deployment_name:
                        del expected_headers["azureml-model-deployment"]
                assert config_output['inference_endpoint']['url'] == expected_scoring_url
                assert config_output['output_settings']['save_partitioned_scoring_results'] == True
                assert config_output['output_settings']['ensure_ascii'] == ensure_ascii
                assert config_output['log_settings']['stdout_log_level'] == ('debug' if debug_mode else 'info')
                assert config_output['request_settings']['timeout'] == (max_retry_time_interval or 0)
                assert config_output['request_settings']['headers'] == expected_headers
