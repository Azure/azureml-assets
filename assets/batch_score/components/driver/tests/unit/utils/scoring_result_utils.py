# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the test utils for scoring result."""

import json

from src.batch_score.common.scoring.http_scoring_response import HttpScoringResponse
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus
)


def assert_scoring_result(
    scoring_result: ScoringResult,
    expected_status: ScoringResultStatus,
    scoring_request: ScoringRequest,
    http_response: HttpScoringResponse,
    end_time: float,
    start_time: float
):
    assert scoring_result.end == end_time
    assert scoring_result.num_retries == 0
    assert scoring_result.status == expected_status
    assert scoring_result.start == start_time
    assert scoring_result.request_metadata == scoring_request.request_metadata
    assert scoring_result.request_obj == json.loads(scoring_request.original_payload)
    assert scoring_result.response_body == http_response.payload
    assert scoring_result.response_headers == http_response.headers