# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for content type header provider."""

from src.batch_score.root.common.header_providers.content_type_header_provider import (
    ContentTypeHeaderProvider,
)


def test_get_headers():
    """Test get_headers method."""
    assert ContentTypeHeaderProvider().get_headers() == {'Content-Type': 'application/json'}
