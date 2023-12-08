# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock tally failed request handler."""

import pytest

from src.batch_score.common.scoring.tally_failed_request_handler import (
    TallyFailedRequestHandler,
)


@pytest.fixture()
def make_tally_failed_request_handler():
    """Mock tally failed request handler."""
    def make(enabled: bool = False, tally_exclusions: str = None) -> TallyFailedRequestHandler:
        """Make a mock tally failed request handler."""
        return TallyFailedRequestHandler(
            enabled=enabled,
            tally_exclusions=tally_exclusions
        )

    return make
