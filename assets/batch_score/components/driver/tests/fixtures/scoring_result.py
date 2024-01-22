import time

import pytest
from multidict import CIMultiDictProxy

from src.batch_score.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus,
)


@pytest.fixture
def make_scoring_result():

    def make(
            status: ScoringResultStatus = ScoringResultStatus.SUCCESS,
            start: float = time.time() - 10,
            end: float = time.time(),
            request_obj: any = None,
            request_metadata: any = None,
            response_body: any = {"usage": {}},
            response_headers: CIMultiDictProxy[str] = None,
            num_retries: int = 0,
            omit: bool = False):
        return ScoringResult(
            status=status,
            start=start,
            end=end,
            request_obj=request_obj,
            request_metadata=request_metadata,
            response_body=response_body,
            response_headers=response_headers,
            num_retries=num_retries,
            omit=omit)

    return make


def get_test_request_obj():
    return {
        "payload": "test request"
    }
