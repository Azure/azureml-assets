# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from mock import patch

from src.batch_score.common.parallel import adjustment
from src.batch_score.common.parallel.adjustment import AIMD


@pytest.fixture()
@patch.object(adjustment, "get_logger")
def make_AIMD(mock_get_logger):
    # TODO: request_metrics=make_request_metrics()
    def make(request_metrics=None):
        return AIMD(request_metrics=request_metrics)

    return make
