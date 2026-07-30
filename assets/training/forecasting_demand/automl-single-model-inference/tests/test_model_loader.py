# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for signed AutoML forecasting model loading."""

import hashlib
import pickle
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils


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


@pytest.fixture
def signing_key(monkeypatch):
    """Create an RSA signing key and configure its public key as trusted."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    monkeypatch.setenv(
        model_loader.MODEL_SIGNING_PUBLIC_KEY_ENV_VAR,
        public_key_pem.decode("utf-8"),
    )
    return private_key


def _sign_model(model_path, private_key):
    digest = hashlib.sha256(model_path.read_bytes()).digest()
    signature = private_key.sign(
        digest,
        padding.PKCS1v15(),
        utils.Prehashed(hashes.SHA256()),
    )
    Path(f"{model_path}{model_loader.SIGNATURE_FILE_POSTFIX}").write_bytes(signature)


def test_get_model_loads_valid_signed_pickle(tmp_path, signing_key):
    """Load a pickle only when its detached signature is valid."""
    model_path = tmp_path / "model.pkl"
    expected_model = {"model": "trusted"}
    model_path.write_bytes(pickle.dumps(expected_model))
    _sign_model(model_path, signing_key)

    assert model_loader.get_model(str(model_path)) == expected_model


def test_get_model_rejects_unsigned_pickle_without_deserializing(tmp_path):
    """Reject an unsigned pickle without executing its payload."""
    global EXPLOIT_EXECUTED
    EXPLOIT_EXECUTED = False
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.raises(ValueError, match="signature file not found"):
        model_loader.get_model(str(model_path))

    assert not EXPLOIT_EXECUTED


def test_get_model_rejects_invalid_signature_without_deserializing(
    tmp_path, signing_key
):
    """Reject a model changed after signing without executing its payload."""
    global EXPLOIT_EXECUTED
    EXPLOIT_EXECUTED = False
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps({"model": "trusted"}))
    _sign_model(model_path, signing_key)
    model_path.write_bytes(pickle.dumps(_MaliciousPayload()))

    with pytest.raises(ValueError, match="signature verification failed"):
        model_loader.get_model(str(model_path))

    assert not EXPLOIT_EXECUTED


def test_get_model_requires_trusted_public_key(tmp_path, monkeypatch):
    """Reject a signed model when no trusted public key is configured."""
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps({"model": "unsigned"}))
    Path(f"{model_path}{model_loader.SIGNATURE_FILE_POSTFIX}").write_bytes(b"signature")
    monkeypatch.delenv(model_loader.MODEL_SIGNING_PUBLIC_KEY_ENV_VAR, raising=False)

    with pytest.raises(
        ValueError, match=model_loader.MODEL_SIGNING_PUBLIC_KEY_ENV_VAR
    ):
        model_loader.get_model(str(model_path))


def test_get_model_rejects_oversized_signature(tmp_path, signing_key):
    """Reject a signature whose size does not match the trusted RSA key."""
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(pickle.dumps({"model": "trusted"}))
    signature_path = Path(
        f"{model_path}{model_loader.SIGNATURE_FILE_POSTFIX}"
    )
    signature_path.write_bytes(b"x" * 257)

    with pytest.raises(ValueError, match="exactly 256 bytes"):
        model_loader.get_model(str(model_path))
