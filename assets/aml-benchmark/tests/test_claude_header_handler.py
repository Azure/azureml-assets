# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ClaudeHeaderHandler."""
import sys
import unittest

from unittest.mock import Mock
from .test_utils import get_src_dir


sys.path.append(get_src_dir())
print(get_src_dir())
from aml_benchmark.batch_benchmark_score.batch_score.header_handlers.claude import (  # noqa: E402
    claude_header_handler)
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel  # noqa: E402


class TestClaudeHeaderHandler(unittest.TestCase):
    """Tests for ClaudeHeaderHandler."""

    def test_claude_header_handler(self):
        """Test that ClaudeHeaderHandler is created with the correct fields."""
        model = OnlineEndpointModel(
            model=None,
            model_version=None,
            model_type=None,
            endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2:1/invoke'
        )
        handler = claude_header_handler.ClaudeHeaderHandler(
            token_provider=Mock(),
            online_endpoint_model=model
        )
        self.assertEqual(handler._aws_region, 'us-east-1')
        self.assertEqual(handler._model_identifier, 'anthropic.claude-v2:1')


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
