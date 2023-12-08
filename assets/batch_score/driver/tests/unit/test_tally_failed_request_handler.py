# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for tally failed request handler."""

import pytest

from src.batch_score.common.scoring.tally_failed_request_handler import TallyFailedRequestHandler

INVALID_SETUP_TESTS = [
    [True, "none|bad_request_in_model"],
]


@pytest.mark.parametrize("enabled, exclusion_str", INVALID_SETUP_TESTS)
def test_invalid_setup(enabled: bool, exclusion_str: str, make_tally_failed_request_handler):
    """Test invalid setup."""
    with pytest.raises(Exception):
        make_tally_failed_request_handler(enabled=enabled, tally_exclusions=exclusion_str)


VALID_SETUP_TESTS = [
    [False, "none|bad_request_in_model"],
    [True, "bad_request_in_model"],
    [True, "none"],
]


@pytest.mark.parametrize("enabled, exclusion_str", VALID_SETUP_TESTS)
def test_valid_setup(enabled: bool, exclusion_str: str, make_tally_failed_request_handler):
    """Test valid setup."""
    make_tally_failed_request_handler(enabled=enabled, tally_exclusions=exclusion_str)


DISABLED_HANDLER_TESTS = [
    ["none", 424, 400],
    ["bad_request_to_model", 424, 400],
]


@pytest.mark.parametrize("exclusion_str, response_status, model_response_status", DISABLED_HANDLER_TESTS)
def test_disabled_handler(mock_get_logger,
                          exclusion_str: str,
                          response_status: int,
                          model_response_status: int,
                          make_tally_failed_request_handler):
    """Test disabled handler."""
    handler = make_tally_failed_request_handler(enabled=False, tally_exclusions=exclusion_str)

    assert not handler.should_tally(response_status=response_status, model_response_status=model_response_status)


ENABLED_HANDLER_NO_TALLY_TESTS = [
    ["bad_request_to_model", 424, 400],
]


@pytest.mark.parametrize("exclusion_str, response_status, model_response_status", ENABLED_HANDLER_NO_TALLY_TESTS)
def test_enabled_handler_no_tally(mock_get_logger,
                                  exclusion_str: str,
                                  response_status: int,
                                  model_response_status: int,
                                  make_tally_failed_request_handler):
    """Test enabled handler no tally."""
    handler = make_tally_failed_request_handler(enabled=True, tally_exclusions=exclusion_str)

    assert not handler.should_tally(response_status=response_status, model_response_status=model_response_status)
    assert mock_get_logger.debug.called or mock_get_logger.info.called


ENABLED_HANDLER_TALLY_TESTS = [
    ["none", 424, 400],
    ["bad_request_to_model", 424, 429],
    ["bad_request_to_model", 424, 500],
    ["bad_request_to_model", 424, None],
    ["bad_request_to_model", 503, None],
]


@pytest.mark.parametrize("exclusion_str, response_status, model_response_status", ENABLED_HANDLER_TALLY_TESTS)
def test_enabled_handler_tally(mock_get_logger,
                               exclusion_str: str,
                               response_status: int,
                               model_response_status: int,
                               make_tally_failed_request_handler):
    """Test enabled handler tally."""
    handler = make_tally_failed_request_handler(enabled=True, tally_exclusions=exclusion_str)

    assert handler.should_tally(response_status=response_status, model_response_status=model_response_status)
    assert mock_get_logger.debug.called or mock_get_logger.info.called


CATEGORIZE_TESTS = [
    ["bad_request_to_model", 424, 400],
    [None, 424, 429],
    [None, 424, 500],
    [None, 424, None],
    [None, 503, None],
]


@pytest.mark.parametrize("expected_category, response_status, model_response_status", CATEGORIZE_TESTS)
def test_categorize(mock_get_logger, expected_category: str, response_status: int, model_response_status: int):
    """Test categorize."""
    assert TallyFailedRequestHandler._categorize(response_status=response_status,
                                                 model_response_status=model_response_status) == expected_category

    assert mock_get_logger.debug.called or mock_get_logger.info.called

    if expected_category and mock_get_logger.debug.called:
        assert expected_category in mock_get_logger.debug.call_args.args[0]
    elif expected_category and mock_get_logger.info.called:
        assert expected_category in mock_get_logger.info.call_args.args[0]
