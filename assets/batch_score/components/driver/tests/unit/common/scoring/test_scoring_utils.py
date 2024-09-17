# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for scoring utils."""

from enum import Enum

import pytest

import src.batch_score_oss.common.scoring.scoring_utils as scoring_utils
from src.batch_score_oss.common.scoring.scoring_utils import RetriableType

CLASSIFICATION_TESTS = [
    [200, None, "", "", RetriableType.NOT_RETRIABLE],
    [400, None, "", "", RetriableType.NOT_RETRIABLE],
    [403, None, "", "", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [404, "Specified traffic group could not be found", "", "", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, "no healthy upstream", "", "model_not_ready", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, "no healthy upstream", "", "too_few_model_instance", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, "no healthy upstream", "None", "model_not_ready", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, "no healthy upstream", None, "model_not_ready", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, "upstream connect error or disconnect/reset before headers", "None", "model_not_ready",
        RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, None, "408", "", RetriableType.RETRY_ON_SAME_ENDPOINT],
    [424, None, "429", "", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
    [424, None, "500", "", RetriableType.RETRY_UNTIL_MAX_RETRIES],
    [424, None, "504", "", RetriableType.RETRY_UNTIL_MAX_RETRIES],
    [429, None, "", "", RetriableType.RETRY_ON_DIFFERENT_ENDPOINT],
]


@pytest.mark.parametrize("response_status, response_payload, model_response_code, "
                         "model_response_reason, expected_classification", CLASSIFICATION_TESTS)
def test_classify_response(response_status: int,
                           response_payload: any,
                           model_response_code: str,
                           model_response_reason: str,
                           expected_classification: Enum):
    """Test classify response."""
    classification = scoring_utils.get_retriable_type(response_status=response_status,
                                                      response_payload=response_payload,
                                                      model_response_code=model_response_code,
                                                      model_response_reason=model_response_reason)

    assert classification == expected_classification
