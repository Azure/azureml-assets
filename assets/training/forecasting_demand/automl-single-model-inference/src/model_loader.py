# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Safely discover and load AutoML forecasting model artifacts."""

import os

import skops.io as skops_io

try:
    import torch

    _torch_present = True
except ImportError:
    torch = None
    _torch_present = False


SKOPS_FILE_POSTFIX = ".skops"
PTH_FILE_POSTFIX = ".pth"
PT_FILE_POSTFIX = ".pt"
_SUPPORTED_MODEL_POSTFIXES = (
    SKOPS_FILE_POSTFIX,
    PT_FILE_POSTFIX,
    PTH_FILE_POSTFIX,
)


def find_model(model_path):
    """Return a supported safe model artifact from a model directory."""
    for postfix in _SUPPORTED_MODEL_POSTFIXES:
        for root, _, files in os.walk(model_path):
            for filename in sorted(files):
                if filename.lower().endswith(postfix):
                    return os.path.join(root, filename)

    raise ValueError(
        f"Unable to find a supported safe model in folder {model_path}. "
        f"Supported formats: {', '.join(_SUPPORTED_MODEL_POSTFIXES)}."
    )


def _map_location_cuda(storage, loc):
    return storage.cuda()


def _load_skops_model(model_full_path):
    untrusted_types = skops_io.get_untrusted_types(file=model_full_path)
    if untrusted_types:
        raise ValueError(
            "The skops model contains types that are not trusted by default: "
            f"{', '.join(sorted(untrusted_types))}."
        )

    return skops_io.load(model_full_path, trusted=[])


def _load_pytorch_model(model_full_path):
    if not _torch_present:
        raise RuntimeError(
            "Loading a Forecasting TCN model requires torch to be installed."
        )

    map_location = _map_location_cuda if torch.cuda.is_available() else "cpu"
    with open(model_full_path, "rb") as model_file:
        return torch.load(
            model_file,
            map_location=map_location,
            weights_only=True,
        )


def get_model(model_full_path):
    """Load a model without allowing arbitrary Python object construction."""
    model_postfix = os.path.splitext(model_full_path)[1].lower()
    print(f"Loading the model from path: {model_full_path}")

    if model_postfix == SKOPS_FILE_POSTFIX:
        fitted_model = _load_skops_model(model_full_path)
    elif model_postfix in (PT_FILE_POSTFIX, PTH_FILE_POSTFIX):
        fitted_model = _load_pytorch_model(model_full_path)
    else:
        raise ValueError(
            f"Unsupported model format '{model_postfix}'. "
            f"Supported formats: {', '.join(_SUPPORTED_MODEL_POSTFIXES)}."
        )

    print("Model loading succeeded.")
    return fitted_model
