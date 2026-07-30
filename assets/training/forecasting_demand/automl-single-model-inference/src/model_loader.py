# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load AutoML forecasting models only after verifying their provenance."""

import hashlib
import os
import pickle
import tempfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils

try:
    import torch

    _torch_present = True
except ImportError:
    _torch_present = False


MODEL_SIGNING_PUBLIC_KEY_ENV_VAR = "AUTOML_MODEL_SIGNING_PUBLIC_KEY_PEM"
PTH_FILE_POSTFIX = ".pth"
PT_FILE_POSTFIX = ".pt"
PICKLE_FILE_POSTFIX = ".pkl"
SIGNATURE_FILE_POSTFIX = ".sig"
_SUPPORTED_MODEL_POSTFIXES = (PICKLE_FILE_POSTFIX, PT_FILE_POSTFIX, PTH_FILE_POSTFIX)
_HASH_CHUNK_SIZE = 1024 * 1024


def _load_signing_public_key():
    public_key_pem = os.environ.get(MODEL_SIGNING_PUBLIC_KEY_ENV_VAR)
    if not public_key_pem:
        raise ValueError(
            f"{MODEL_SIGNING_PUBLIC_KEY_ENV_VAR} must contain the trusted RSA public key "
            "used to verify model artifacts."
        )

    try:
        public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    except (TypeError, ValueError) as ex:
        raise ValueError(
            f"{MODEL_SIGNING_PUBLIC_KEY_ENV_VAR} does not contain a valid PEM public key."
        ) from ex

    if not isinstance(public_key, rsa.RSAPublicKey):
        raise ValueError("The model-signing public key must be an RSA public key.")

    return public_key


def _stage_and_verify_model(model_full_path):
    signature_path = f"{model_full_path}{SIGNATURE_FILE_POSTFIX}"
    if not os.path.isfile(signature_path):
        raise ValueError(f"Model signature file not found: {signature_path}")

    public_key = _load_signing_public_key()
    hasher = hashlib.sha256()
    staged_path = None
    verified = False

    try:
        model_postfix = os.path.splitext(model_full_path)[1].lower()
        with open(model_full_path, "rb") as model_file, tempfile.NamedTemporaryFile(
            mode="wb", suffix=model_postfix, delete=False
        ) as staged_file:
            staged_path = staged_file.name
            while chunk := model_file.read(_HASH_CHUNK_SIZE):
                hasher.update(chunk)
                staged_file.write(chunk)

        expected_signature_size = (public_key.key_size + 7) // 8
        with open(signature_path, "rb") as signature_file:
            signature = signature_file.read(expected_signature_size + 1)
        if len(signature) != expected_signature_size:
            raise ValueError(
                f"Model signature must be exactly {expected_signature_size} bytes."
            )

        public_key.verify(
            signature,
            hasher.digest(),
            padding.PKCS1v15(),
            utils.Prehashed(hashes.SHA256()),
        )
        verified = True
        return staged_path
    except InvalidSignature as ex:
        raise ValueError(f"Model signature verification failed for {model_full_path}.") from ex
    finally:
        if staged_path and not verified:
            os.unlink(staged_path)


def _map_location_cuda(storage, loc):
    return storage.cuda()


def get_model(model_full_path):
    """Verify and load a supported AutoML forecasting model."""
    model_postfix = os.path.splitext(model_full_path)[1].lower()
    if model_postfix not in _SUPPORTED_MODEL_POSTFIXES:
        raise ValueError(
            f"Unsupported model format '{model_postfix}'. "
            f"Supported formats: {', '.join(_SUPPORTED_MODEL_POSTFIXES)}."
        )

    print(f"Verifying the model signature for path: {model_full_path}")
    staged_model_path = _stage_and_verify_model(model_full_path)
    try:
        print(f"Loading the verified model from path: {model_full_path}")
        # Legacy AutoML artifacts require full-object loading; the detached signature is the trust boundary.
        if model_postfix in (PT_FILE_POSTFIX, PTH_FILE_POSTFIX):
            if not _torch_present:
                raise RuntimeError(
                    "Loading Forecasting TCN model requires torch to be installed in the environment."
                )

            map_location = _map_location_cuda if torch.cuda.is_available() else "cpu"
            with open(staged_model_path, "rb") as model_file:
                fitted_model = torch.load(
                    model_file,
                    map_location=map_location,
                    weights_only=False,
                )
        else:
            with open(staged_model_path, "rb") as model_file:
                fitted_model = pickle.load(model_file)
    finally:
        os.unlink(staged_model_path)

    print("Model loading succeeded.")
    return fitted_model
