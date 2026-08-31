# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for safe AutoML forecasting model loading."""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as skops_io
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
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


class _OtherUntrustedModel:
    pass


def _symlink_or_skip(link_path, target_path, target_is_directory=False):
    try:
        link_path.symlink_to(target_path, target_is_directory=target_is_directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"Symbolic links are unavailable: {error}")


def test_get_model_loads_trusted_skops_model(tmp_path):
    """Load a trusted regression pipeline without changing its predictions."""
    model_path = tmp_path / "model.skops"
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([1.0, 3.0, 5.0, 7.0])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    ).fit(features, labels)
    skops_io.dump(model, model_path)

    loaded_model = model_loader.get_model(str(model_path))

    assert loaded_model.predict(features) == pytest.approx(model.predict(features))


def test_get_model_rejects_untrusted_skops_types(tmp_path):
    """Reject skops artifacts containing custom untrusted types."""
    model_path = tmp_path / "model.skops"
    skops_io.dump(_UntrustedModel(), model_path)

    with pytest.raises(ValueError, match="not explicitly trusted"):
        model_loader.get_model(str(model_path))


def test_get_model_passes_only_explicitly_allowlisted_types(tmp_path, monkeypatch):
    """Pass exact allowlisted types to skops instead of enabling broad trust."""
    model_path = tmp_path / "model.skops"
    skops_io.dump(_UntrustedModel(), model_path)
    untrusted_types = skops_io.get_untrusted_types(file=model_path)
    assert len(untrusted_types) == 1
    monkeypatch.setattr(
        model_loader,
        "_TRUSTED_AUTOML_FORECASTING_TYPES",
        frozenset(untrusted_types),
    )

    loaded_model = model_loader.get_model(str(model_path))

    assert isinstance(loaded_model, _UntrustedModel)


def test_get_model_rejects_types_outside_explicit_allowlist(tmp_path, monkeypatch):
    """Reject an artifact when even one custom type is not allowlisted."""
    model_path = tmp_path / "model.skops"
    skops_io.dump([_UntrustedModel(), _OtherUntrustedModel()], model_path)
    untrusted_types = skops_io.get_untrusted_types(file=model_path)
    allowed_type = next(name for name in untrusted_types if name.endswith("._UntrustedModel"))
    monkeypatch.setattr(
        model_loader,
        "_TRUSTED_AUTOML_FORECASTING_TYPES",
        frozenset({allowed_type}),
    )

    with pytest.raises(ValueError, match="_OtherUntrustedModel"):
        model_loader.get_model(str(model_path))


@pytest.mark.parametrize(
    "offset",
    [pd.offsets.Day(), pd.offsets.Week(), pd.offsets.MonthEnd(), pd.offsets.Minute()],
)
def test_get_model_loads_supported_forecasting_frequencies(tmp_path, offset):
    """Load common non-hourly forecasting frequencies by exact type."""
    model_path = tmp_path / "model.skops"
    skops_io.dump(offset, model_path)

    loaded_offset = model_loader.get_model(str(model_path))

    assert loaded_offset == offset


def test_get_model_rejects_malicious_pickle_without_deserializing(tmp_path):
    """Reject a pickle disguised as a skops model without executing it."""
    global EXPLOIT_EXECUTED
    EXPLOIT_EXECUTED = False
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.raises(ValueError, match="not a valid ZIP archive"):
        model_loader.get_model(str(model_path))

    assert not EXPLOIT_EXECUTED


def test_find_model_selects_safe_model_alongside_legacy_model(tmp_path):
    """Select the safe sibling while retaining backward-compatible artifacts."""
    (tmp_path / "model.pkl").write_bytes(b"pickle")
    safe_model = tmp_path / "model.skops"
    safe_model.write_bytes(b"skops")

    assert model_loader.find_model(str(tmp_path)) == str(safe_model)


def test_find_model_rejects_directory_with_only_pickle(tmp_path):
    """Reject a model directory that contains only an unsafe pickle."""
    (tmp_path / "model.pkl").write_bytes(b"pickle")

    with pytest.raises(ValueError, match="Unsafe legacy model artifacts"):
        model_loader.find_model(str(tmp_path))


@pytest.mark.parametrize("postfix", [".pt", ".pth"])
def test_find_model_rejects_legacy_pytorch_artifacts(tmp_path, postfix):
    """Reject full-object PyTorch models until trusted reconstruction exists."""
    (tmp_path / f"model{postfix}").write_bytes(b"pytorch")

    with pytest.raises(ValueError, match="Unsafe legacy model artifacts"):
        model_loader.find_model(str(tmp_path))


def test_find_model_rejects_ambiguous_safe_models(tmp_path):
    """Reject directories containing more than one candidate model."""
    (tmp_path / "first.skops").write_bytes(b"first")
    (tmp_path / "second.skops").write_bytes(b"second")

    with pytest.raises(ValueError, match="Expected exactly one"):
        model_loader.find_model(str(tmp_path))


def test_get_model_rejects_unsupported_extension(tmp_path):
    """Reject direct attempts to load a legacy model format."""
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"pickle")

    with pytest.raises(ValueError, match="Unsupported model format"):
        model_loader.get_model(str(model_path))


def test_find_model_rejects_non_directory_path(tmp_path):
    """Reject a file passed where a model directory is required."""
    model_path = tmp_path / "model.skops"
    model_path.write_bytes(b"skops")

    with pytest.raises(ValueError, match="regular directory"):
        model_loader.find_model(str(model_path))


def test_find_model_rejects_directory_symlink(tmp_path):
    """Reject a model path that redirects to another directory."""
    real_model_directory = tmp_path / "real-model"
    real_model_directory.mkdir()
    (real_model_directory / "model.skops").write_bytes(b"skops")
    linked_model_directory = tmp_path / "linked-model"
    _symlink_or_skip(
        linked_model_directory,
        real_model_directory,
        target_is_directory=True,
    )

    with pytest.raises(ValueError, match="regular directory"):
        model_loader.find_model(str(linked_model_directory))


def test_find_model_rejects_skops_symlink_in_directory(tmp_path):
    """Reject a .skops artifact that redirects to another file."""
    real_model = tmp_path / "real-model.skops"
    real_model.write_bytes(b"skops")
    model_directory = tmp_path / "model-directory"
    model_directory.mkdir()
    linked_model = model_directory / "model.skops"
    _symlink_or_skip(linked_model, real_model)

    with pytest.raises(ValueError, match="Symbolic links"):
        model_loader.find_model(str(model_directory))
