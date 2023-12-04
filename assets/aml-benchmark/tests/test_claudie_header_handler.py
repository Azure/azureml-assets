# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ClaudieHeaderHandler"""

import unittest

from unittest.mock import Mock

from aml_benchmark.batch_benchmark_score.batch_score.header_handlers.claudie.claudie_header_handler import ClaudieHeaderHandler
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel



class TestClaudieHeaderHandler(unittest.TestCase):
    """Tests for ClaudieHeaderHandler"""

    def test_claudie_header_handler(self):
        """Test that ClaudieHeaderHandler is created with the correct fields."""
        model = OnlineEndpointModel(
            model=None,
            model_version=None,
            model_type=None,
            endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2.1/invoke'
            )
        handler = ClaudieHeaderHandler(
            token_provider=Mock(),
            online_endpoint_model=model
            )
        self.assertEqual(handler._aws_region, 'us-east-1')
        self.assertEqual(handler._model_identifier, 'anthropic.claude-v2.1')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()