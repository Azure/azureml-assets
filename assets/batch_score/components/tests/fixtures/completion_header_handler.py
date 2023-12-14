# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock completion header handler."""

import pytest

from src.batch_score.header_handlers.open_ai.completion_header_handler import (
    CompletionHeaderHandler,
)


@pytest.fixture
def make_completion_header_handler(make_token_provider):
    """Make a mock completion header handler."""
    def make(token_provider=make_token_provider(),
             user_agent_segment=None,
             batch_pool="TEST_POOL",
             quota_audience="TEST_AUDIENCE"):
        return CompletionHeaderHandler(
            token_provider=token_provider,
            user_agent_segment=user_agent_segment,
            batch_pool=batch_pool,
            quota_audience=quota_audience,
        )

    return make
