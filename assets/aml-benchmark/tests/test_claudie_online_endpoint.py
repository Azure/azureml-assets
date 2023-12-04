# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for claudie endpoint."""
import os
import unittest

from ddt import ddt, data

from aml_benchmark.utils.online_endpoint.claudie_online_endpoint import ClaudieOnlineEndpoint
from aml_benchmark.batch_benchmark_score.batch_score.utils.exceptions import BenchmarkUserException
from aml_benchmark.batch_benchmark_score.batch_score.utils.error_definitions import BenchmarkUserError
from unittest.mock import patch


@ddt
class TestClaudieOnlineEndpoint(unittest.TestCase):

    def setUp(self):
        os.environ[ClaudieOnlineEndpoint.ACCESS_KEY] = 'MockAccessKey'
        os.environ[ClaudieOnlineEndpoint.SECRET_KEY] = 'SuperSecretMockKey'
        unittest.TestCase.setUp(self)

    def test_no_region_raises(self):
        """Test that the exception is raised if no region is set."""
        with self.assertRaises(BenchmarkUserException) as cm:
            ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region=None,
                model_identifier='anthropic.claude-v2',
                payload='foo')
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertEqual(cm.exception.args[0], 'Please provide the aws_region parameter.')

    def test_no_model_raises(self):
        """Test that the exception is raised if no region is set."""
        with self.assertRaises(BenchmarkUserException) as cm:
            ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region='us-east-1',
                model_identifier=None,
                payload='foo')
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertEqual(cm.exception.args[0], 'Please provide the model_identifier parameter.')

    def test_no_payload_raises(self):
        """Test that the exception is raised if no region is set."""
        with self.assertRaises(BenchmarkUserException) as cm:
            ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2')
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertEqual(cm.exception.args[0], 'Please provide the payload parameter.')

    @data([ClaudieOnlineEndpoint.ACCESS_KEY],
          [ClaudieOnlineEndpoint.SECRET_KEY],
          [ClaudieOnlineEndpoint.ACCESS_KEY, ClaudieOnlineEndpoint.SECRET_KEY]
          )
    def test_no_key(self, keys_to_delete):
        """Assert that the exception is raised if the keys were not provided."""
        for k in keys_to_delete:
            del os.environ[k]
        with self.assertRaises(BenchmarkUserException) as cm:
            ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload='foo')
        self.assertEqual(cm.exception._azureml_error.error_definition.code,
                         BenchmarkUserError().code)
        self.assertIn(ClaudieOnlineEndpoint.ACCESS_KEY, cm.exception.args[0])
        self.assertIn(ClaudieOnlineEndpoint.SECRET_KEY, cm.exception.args[0])

    def test_scoring_url(self):
        """Check that the returned url is correct."""
        endpoint = ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload='foo')
        self.assertEqual(endpoint.scoring_url,
                         'https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke')

    def test_hex_digest(self):
        """Test that we are correcly calculating the hexdigest of a payload."""
        endpoint = ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
                aws_region='us-east-1',
                model_identifier='anthropic.claude-v2',
                payload=('{"prompt":"\\n\\nHuman:story of two dogs\\n'
                         '\\nAssistant:","max_tokens_to_sample":100}'))
        self.assertEqual(endpoint.payload_hash,
                         '09b8e66f6aea38a57a8cf0dd388f168f4c76a9c9e5b63afbed46c1553613211b')

    def test_canonical_headers(self):
        """Test that canonical headers are being returned correctly."""
        endpoint = ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
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
        canonical_headers_str, signed_headers_str = endpoint._get_canonical_header_string_and_signed_headers(expected_headers)
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
            'aml_benchmark.utils.online_endpoint.claudie_online_endpoint.ClaudieOnlineEndpoint._get_date_and_time',
            return_value=('20231201T192721Z', '20231201')):
            endpoint = ClaudieOnlineEndpoint(
                workspace_name='mock_ws',
                resource_group='mock_rg',
                subscription_id='mock_subscription',
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