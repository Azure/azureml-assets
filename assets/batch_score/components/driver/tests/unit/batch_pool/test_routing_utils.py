# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for routing utils."""

import pytest

import src.batch_score.batch_pool.routing.routing_utils as routing_utils

CLASSIFICATION_TESTS = [
    [200, routing_utils.RoutingResponseType.SUCCESS],
    [408, routing_utils.RoutingResponseType.RETRY],
    [-408, routing_utils.RoutingResponseType.RETRY],
    [500, routing_utils.RoutingResponseType.USE_EXISTING],
    [404, routing_utils.RoutingResponseType.FAILURE],
]


@pytest.mark.parametrize("response_status, expected_classification", CLASSIFICATION_TESTS)
def test_classify_response(mock_get_logger,
                           response_status: int,
                           expected_classification: routing_utils.RoutingResponseType):
    """Test classify response."""
    classification = routing_utils.classify_response(response_status=response_status)

    assert mock_get_logger.info.called or mock_get_logger.debug.called
    assert classification == expected_classification

    if mock_get_logger.info.called:
        assert str(response_status) in mock_get_logger.info.call_args.args[0] and \
                str(expected_classification) in mock_get_logger.info.call_args.args[0]
    else:  # mock_get_logger.debug.called
        assert str(response_status) in mock_get_logger.debug.call_args.args[0] and \
                str(expected_classification) in mock_get_logger.debug.call_args.args[0]
