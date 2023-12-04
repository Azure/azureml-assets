# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for OnlineEndpointModel"""

import unittest
from ddt import ddt, data, unpack
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel


@ddt
class TestOnlineEndpointModel(unittest.TestCase):
    """Tests for OnlineEndpointModel"""

    @data(['https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/invoke', '2'],
          ['https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2.1/invoke', '2.1'])
    @unpack
    def test_claudie_endpoint(self, path, expected_version):
        """Test that claudie endpoint has the correct name."""
        claudie_model = OnlineEndpointModel(
            model=None,
            model_version=None,
            model_type=None,
            endpoint_url=path
        )
        self.assertEqual(claudie_model.model_name, 'anthropic.claude')
        self.assertEqual(claudie_model.model_path, path)
        self.assertEqual(claudie_model.model_version, expected_version)
        self.assertEqual(claudie_model.model_type, 'claude')
        self.assertTrue(claudie_model.is_claude_model())


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
