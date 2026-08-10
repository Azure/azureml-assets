# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Safely discover and load AutoML forecasting model artifacts."""

import os
import zipfile

import skops.io as skops_io


SKOPS_FILE_POSTFIX = ".skops"
_LEGACY_MODEL_POSTFIXES = (".pkl", ".pickle", ".pt", ".pth")


def find_model(model_path):
    """Return the single supported model artifact from a model directory."""
    if not os.path.isdir(model_path) or os.path.islink(model_path):
        raise ValueError("The model path must be a regular directory.")

    safe_models = []
    legacy_models = []
    for root, _, files in os.walk(model_path):
        for filename in sorted(files):
            model_full_path = os.path.join(root, filename)
            postfix = os.path.splitext(filename)[1].lower()
            if postfix == SKOPS_FILE_POSTFIX:
                safe_models.append(model_full_path)
            elif postfix in _LEGACY_MODEL_POSTFIXES:
                legacy_models.append(model_full_path)

    if not safe_models and legacy_models:
        raise ValueError(
            "Unsafe legacy model artifacts are not supported by this component: "
            f"{', '.join(legacy_models)}. Export the fitted forecasting model "
            "directly to .skops without loading or converting an untrusted pickle."
        )

    if len(safe_models) > 1:
        raise ValueError(
            "Expected exactly one .skops model artifact, but found "
            f"{len(safe_models)}: {', '.join(safe_models)}."
        )

    if safe_models:
        model_full_path = safe_models[0]
        if os.path.islink(model_full_path):
            raise ValueError("Symbolic links are not accepted as model artifacts.")
        return model_full_path

    raise ValueError(
        f"Unable to find a supported safe model in folder {model_path}. "
        f"Supported format: {SKOPS_FILE_POSTFIX}."
    )


def _load_skops_model(model_full_path):
    if not os.path.isfile(model_full_path) or os.path.islink(model_full_path):
        raise ValueError("The model artifact must be a regular file.")
    if not zipfile.is_zipfile(model_full_path):
        raise ValueError("The .skops model artifact is not a valid ZIP archive.")

    untrusted_types = skops_io.get_untrusted_types(file=model_full_path)
    if untrusted_types:
        raise ValueError(
            "The skops model contains types that are not trusted by default: "
            f"{', '.join(sorted(untrusted_types))}."
        )

    return skops_io.load(model_full_path, trusted=[])


def get_model(model_full_path):
    """Load a model without allowing arbitrary Python object construction."""
    model_postfix = os.path.splitext(model_full_path)[1].lower()
    print(f"Loading the model from path: {model_full_path}")

    if model_postfix == SKOPS_FILE_POSTFIX:
        fitted_model = _load_skops_model(model_full_path)
    else:
        raise ValueError(
            f"Unsupported model format '{model_postfix}'. "
            f"Supported format: {SKOPS_FILE_POSTFIX}."
        )

    print("Model loading succeeded.")
    return fitted_model
