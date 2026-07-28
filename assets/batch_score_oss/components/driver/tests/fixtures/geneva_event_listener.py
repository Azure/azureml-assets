# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock Geneva event listener."""

import importlib
import pytest

from mock import MagicMock

original_import = importlib.import_module


def mock_import(name, *args):
    """Mock the dynamic import process of the telemetry module."""
    if name == "azureml_common.parallel_run.telemetry_logger":
        return MagicMock()
    return original_import(name, *args)


@pytest.fixture
def mock_import_module(monkeypatch):
    """Pytest fixture to mock the dynamic import process of the telemetry module."""
    monkeypatch.setattr("importlib.import_module", mock_import)
