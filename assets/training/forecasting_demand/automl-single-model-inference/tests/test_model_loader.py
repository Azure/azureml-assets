# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for safe AutoML forecasting model loading."""

import pickle
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import skops.io as skops_io
from sklearn.preprocessing import StandardScaler


SRC_PATH = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC_PATH))

import model_loader  # noqa: E402


EXPLOIT_EXECUTED = False


def _mark_executed():
    global EXPLOIT_EXECUTED
    EXPLOIT_EXECUTED = True


class _MaliciousPayload:
    def __reduce__(self):
        return _mark_executed, ()


class _UntrustedModel:
    pass


def test_get_model_loads_trusted_skops_model(tmp_path):
    """Load a skops model containing only trusted sklearn types."""
    model_path = tmp_path / "model.skops"
    model = StandardScaler().fit([[0.0], [1.0], [2.0]])
    skops_io.dump(model, model_path)

    loaded_model = model_loader.get_model(str(model_path))

    assert loaded_model.mean_ == pytest.approx(model.mean_)


def test_get_model_rejects_untrusted_skops_types(tmp_path):
    """Reject skops artifacts containing custom untrusted types."""
    model_path = tmp_path / "model.skops"
    skops_io.dump(_UntrustedModel(), model_path)

    with pytest.raises(ValueError, match="not trusted by default"):
        model_loader.get_model(str(model_path))


def test_get_model_rejects_malicious_pickle_without_deserializing(tmp_path):
    """Reject a pickle disguised as a skops model without executing it."""
    global EXPLOIT_EXECUTED
    EXPLOIT_EXECUTED = False
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.raises(zipfile.BadZipFile):
        model_loader.get_model(str(model_path))

    assert not EXPLOIT_EXECUTED


def test_get_model_uses_weights_only_for_pytorch(tmp_path, monkeypatch):
    """Load PyTorch artifacts with arbitrary object construction disabled."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"weights")
    torch_mock = Mock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.load.return_value = {"weight": 1}
    monkeypatch.setattr(model_loader, "torch", torch_mock)
    monkeypatch.setattr(model_loader, "_torch_present", True)

    assert model_loader.get_model(str(model_path)) == {"weight": 1}
    torch_mock.load.assert_called_once()
    assert torch_mock.load.call_args.kwargs["weights_only"] is True


def test_find_model_ignores_pickle_files(tmp_path):
    """Discover safe formats without falling back to pickle artifacts."""
    (tmp_path / "model.pkl").write_bytes(b"pickle")
    safe_model = tmp_path / "model.skops"
    safe_model.write_bytes(b"skops")

    assert model_loader.find_model(str(tmp_path)) == str(safe_model)


def test_find_model_rejects_directory_with_only_pickle(tmp_path):
    """Reject a model directory that contains only an unsafe pickle."""
    (tmp_path / "model.pkl").write_bytes(b"pickle")

    with pytest.raises(ValueError, match="supported safe model"):
        model_loader.find_model(str(tmp_path))
