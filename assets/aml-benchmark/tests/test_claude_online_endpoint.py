# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for claude endpoint."""
import unittest

from ddt import ddt, data
from unittest.mock import patch

from aml_benchmark.utils.online_endpoint.claude_online_endpoint import ClaudeOnlineEndpoint
from aml_benchmark.batch_benchmark_score.batch_score.utils.exceptions import BenchmarkUserException
from aml_benchmark.batch_benchmark_score.batch_score.utils.error_definitions import BenchmarkUserError


@ddt
class TestClaudeOnlineEndpoint(unittest.TestCase):

    def setUp(self):
        self.returned_cred = {'properties': {'credentials': {'keys': {
            ClaudeOnlineEndpoint.ACCESS_KEY: 'MockAccessKey',
            ClaudeOnlineEndpoint.SECRET_KEY: 'SuperSecretMockKey'
        }}}}
        unittest.TestCase.setUp(self)

    @data(
        'connections_name',
        'aws_region',
        'model_identifier',
        'payload'
    )
    def test_no_param_raises(self, param):
        """Test that the exception is raised if no region is set."""
        kwargs = dict(
            workspace_name='mock_ws',
            resource_group='mock_rg',
            subscription_id='mock_subscription',
            connections_name='mock_connection',
            aws_region='us-east-1',
            model_identifier='anthropic.claude-v2',
            payload='foo')
        kwargs[param] = None
        with self.assertRaises(BenchmarkUserException) as cm:
            ClaudeOnlineEndpoint(**kwargs)
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertEqual(cm.exception.args[0], f'Please provide the {param} parameter.')

    @data([ClaudeOnlineEndpoint.ACCESS_KEY],
          [ClaudeOnlineEndpoint.SECRET_KEY],
          [ClaudeOnlineEndpoint.ACCESS_KEY, ClaudeOnlineEndpoint.SECRET_KEY]
          )
    def test_no_key(self, keys_to_delete):
        """Assert that the exception is raised if the keys were not provided."""
        for k in keys_to_delete:
            del self.returned_cred['properties']['credentials']['keys'][k]
        with self.assertRaises(BenchmarkUserException) as cm:
            with patch(
                ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
                 '.ClaudeOnlineEndpoint._get_connections_by_name'),
                    return_value=self.returned_cred):
                ClaudeOnlineEndpoint(
                    workspace_name='mock_ws',
                    resource_group='mock_rg',
                    subscription_id='mock_subscription',
                    connections_name='mock_connection',
                    aws_region='us-east-1',
                    model_identifier='anthropic.claude-v2',
                    payload='foo')
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertIn(ClaudeOnlineEndpoint.ACCESS_KEY, cm.exception.args[0])
        self.assertIn(ClaudeOnlineEndpoint.SECRET_KEY, cm.exception.args[0])

    def test_scoring_url(self):
        """Check that the returned url is correct."""
        with patch(
            ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
             '.ClaudeOnlineEndpoint._get_connections_by_name'),
                return_value=self.returned_cred):
            endpoint = ClaudeOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                connections_name='mock_connection',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload='foo')
            self.assertEqual(endpoint.scoring_url,
                             'https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke')

    def test_hex_digest(self):
        """Test that we are correcly calculating the hexdigest of a payload."""
        with patch(
            ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
             '.ClaudeOnlineEndpoint._get_connections_by_name'),
                return_value=self.returned_cred):
            endpoint = ClaudeOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                connections_name='mock_connection',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload=('{"prompt":"\\n\\nHuman:story of two dogs\\n'
                         '\\nAssistant:","max_tokens_to_sample":100}'))
        self.assertEqual(endpoint.payload_hash,
                         '09b8e66f6aea38a57a8cf0dd388f168f4c76a9c9e5b63afbed46c1553613211b')

    def test_canonical_headers(self):
        """Test that canonical headers are being returned correctly."""
        with patch(
            ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
             '.ClaudeOnlineEndpoint._get_connections_by_name'),
                return_value=self.returned_cred):
            endpoint = ClaudeOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                connections_name='mock_connection',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload=('{"prompt":"\\n\\nHuman:story of two dogs\\n'
                         '\\nAssistant:","max_tokens_to_sample":100}'))
        expected_headers = {
            'accept': 'application/json',
            'host': 'bedrock-runtime.us-east-1.amazonaws.com',
            'content-type': 'application/json',
            'X-Amz-Content-Sha256': '09b8e66f6aea38a57a8cf0dd388f168f4c76a9c9e5b63afbed46c1553613211b',
            'X-Amz-Date': '20231201T192721Z'
        }
        self.assertDictEqual(expected_headers, endpoint._get_canonical_headers('20231201T192721Z'))
        canonical_headers_str, signed_headers_str = endpoint._get_canonical_header_string_and_signed_headers(
            expected_headers)
        self.assertEqual(canonical_headers_str,
                         ('accept:application/json\ncontent-type:application/json\n'
                          'host:bedrock-runtime.us-east-1.amazonaws.com\nx-amz-content-sha256'
                          ':09b8e66f6aea38a57a8cf0dd388f168f4c76a9c9e5b63afbed46c1553613211b\n'
                          'x-amz-date:20231201T192721Z\n'))
        self.assertEqual(signed_headers_str,
                         'accept;content-type;host;x-amz-content-sha256;x-amz-date')

    def test_get_endpoint_authorization_header(self):
        """Check the correctness of a signature."""
        with patch(
            ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
             '.ClaudeOnlineEndpoint._get_date_and_time'),
                return_value=('20231201T192721Z', '20231201')):
            with patch(
                ('aml_benchmark.utils.online_endpoint.claude_online_endpoint'
                 '.ClaudeOnlineEndpoint._get_connections_by_name'),
                    return_value=self.returned_cred):
                endpoint = ClaudeOnlineEndpoint(
                    workspace_name='mock_ws',
                    resource_group='mock_rg',
                    subscription_id='mock_subscription',
                    connections_name='mock_connection',
                    aws_region='us-east-1',
                    model_identifier='anthropic.claude-v2',
                    payload=('{"prompt":"\\n\\nHuman:story of two dogs\\n'
                             '\\nAssistant:","max_tokens_to_sample":100}'))
            headers = endpoint.get_endpoint_authorization_header()
        self.assertIn('Authorization', headers.keys())
        self.assertEqual(headers['Authorization'],
                         ('AWS4-HMAC-SHA256 '
                          'Credential=MockAccessKey/20231201/us-east-1/bedrock/aws4_request, '
                          'SignedHeaders=accept;content-type;host;x-amz-content-sha256;x-amz-date, '
                          'Signature=e663b15e1a8dc63fde82afa2c3dc625e57700240147c7e7725a38109b0eb5c1d')
                         )


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
