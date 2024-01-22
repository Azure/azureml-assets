import importlib
import pytest

from mock import MagicMock

original_import = importlib.import_module


def mock_import(name, *args):
    if name == "azureml_common.parallel_run.telemetry_logger":
        return MagicMock()
    return original_import(name, *args)


@pytest.fixture
def mock_import_module(monkeypatch):
    monkeypatch.setattr("importlib.import_module", mock_import)
