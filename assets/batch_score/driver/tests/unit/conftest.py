# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import requests

from src.batch_score.common.telemetry.logging_utils import setup_logger


# Marks all tests in this directory as unit tests
@pytest.fixture(autouse=True, params=[pytest.param(None, marks=pytest.mark.unit)])
def mark_as_unit_test():
    pass


# Sets up the logger for all tests in this directory
@pytest.fixture(autouse=True, params=[pytest.param(None, marks=pytest.mark.unit)])
def setup_logger_fixture():
    setup_logger('DEBUG')


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch):
    def stunted_network_request():
        raise RuntimeError("Network access not allowed during unit testing!")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: stunted_network_request())
    monkeypatch.setattr(requests, "put", lambda *args, **kwargs: stunted_network_request())
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: stunted_network_request())
    monkeypatch.setattr(requests, "patch", lambda *args, **kwargs: stunted_network_request())
    monkeypatch.setattr(requests, "delete", lambda *args, **kwargs: stunted_network_request())
