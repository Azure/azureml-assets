# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Preprocess model."""

import azureml.model.mgmt.processors.hftransformers as hftransformers
from azureml.model.mgmt.config import ModelFlavor
from pathlib import Path


def run_preprocess(model_flavor: str, model_path: Path, output_dir: Path, **preprocess_args):
    """Preprocess model."""
    print(f"Run preprocess for model with flavor: {model_flavor} at path: {model_path}")
    if model_flavor == ModelFlavor.HFTRANSFORMERS.value:
        hftransformers.to_mlflow(model_path, output_dir, preprocess_args)
    else:
        raise Exception(f"Unsupported model flavor: {model_flavor}.")
    print("Model prepocessing completed !!!")
