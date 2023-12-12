# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for parallel driver."""

import json

import pytest

from src.batch_score.common.parallel.parallel_driver import Parallel
from src.batch_score.common.request_modification.modifiers.request_modifier import (
    RequestModificationException,
    RequestModifier,
)


def test_start_with_request_modifier(
        mock_get_logger,
        mock_AIMD,
        mock_run,
        make_parallel_driver,
        make_input_transformer):
    """Test start with request modifier."""
    parallel_driver: Parallel = make_parallel_driver(
        input_to_request_transformer=make_input_transformer(modifiers=[FakeRequestModifier()]))

    payloads = ['{"fake": "payload"}', '{"raise_exception": true}']
    results = parallel_driver.run(payloads=payloads)

    assert len(results) == len(payloads)  # No payloads should be omitted


def test_start_applies_input_to_output_transformer_to_result(
        mock_get_logger,
        mock_AIMD,
        mock_run,
        make_parallel_driver,
        make_input_transformer):
    """Tests the input_to_output_transformer is applied to scoring request objects that are part of the output."""
    parallel_driver: Parallel = make_parallel_driver(
        input_to_output_transformer=make_input_transformer(modifiers=[FakeOutputRequestModifier()]))

    payloads = ['{"raise_exception": true}']
    results = parallel_driver.run(payloads=payloads)

    assert FakeOutputRequestModifier.CHANGED_OUTPUT in results[0]
    assert "true" not in results[0].lower()


def test_start_parses_request_metadata_over_batch_metadata_from_input(
        mock_get_logger,
        mock_AIMD,
        mock_run,
        make_parallel_driver):
    """Test start parses request metadata over batch metadata from input."""
    parallel_driver: Parallel = make_parallel_driver()

    payloads = ['{"prompt": "hey", "request_metadata": "foo", "_batch_request_metadata": "bar"}']

    results = parallel_driver.run(payloads=payloads)

    result = json.loads(results[0])
    assert result["request_metadata"] == "foo"
    assert "_batch_request_metadata" not in result
    assert "bar" not in result


def test_start_parses_batch_metadata_from_input(
        mock_get_logger,
        mock_AIMD,
        mock_run,
        make_parallel_driver):
    """Test start parses batch metadata from input."""
    parallel_driver: Parallel = make_parallel_driver()

    payloads = ['{"prompt": "hey", "_batch_request_metadata": "bar"}']

    results = parallel_driver.run(payloads=payloads)

    result = json.loads(results[0])
    assert result["request_metadata"] == "bar"
    assert "_batch_request_metadata" not in result


def test_start_parses_request_metadata_from_input(
        mock_get_logger,
        mock_AIMD,
        mock_run,
        make_parallel_driver):
    """Test start parses request metadata from input."""
    parallel_driver: Parallel = make_parallel_driver()

    payloads = ['{"prompt": "hey", "request_metadata": "foo"}']

    results = parallel_driver.run(payloads=payloads)

    result = json.loads(results[0])
    assert result["request_metadata"] == "foo"
    assert "_batch_request_metadata" not in result


class FakeRequestModifier(RequestModifier):
    """Mock request modifier."""

    def modify(self, request_obj: any) -> any:
        """Mock modify function."""
        if "raise_exception" in request_obj and request_obj["raise_exception"] is True:
            raise FakeRequestModifierRaiseException()
        else:
            return request_obj


class FakeOutputRequestModifier(RequestModifier):
    """Mock output request modifier."""

    CHANGED_OUTPUT = "<CHANGED OUTPUT>"

    def modify(self, request_obj: any) -> any:
        """Mock modify function."""
        for key in request_obj:
            request_obj[key] = FakeOutputRequestModifier.CHANGED_OUTPUT

        return request_obj


class FakeRequestModifierRaiseException(RequestModificationException):
    """Mock request modifier raise exception."""

    def __init__(self, message: str = "Faked RequestModificationException") -> None:
        """Init function."""
        super().__init__(message)


@pytest.fixture()
def mock_AIMD(monkeypatch):
    """Mock AIMD."""

    class FakeAIMD():
        """Mock AIMD."""

        def __init__(self, *args, **kwargs) -> None:
            """Init function."""
            pass

    monkeypatch.setattr("src.batch_score.common.parallel.adjustment.AIMD.__init__", FakeAIMD.__init__)
