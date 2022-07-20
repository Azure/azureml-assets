# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from multiple libraries.
"""

MODEL_ARCH_MAP = {
    # TorchVision models
    "resnet50": {"input_size": 224, "library": "keras"},
}

MODEL_ARCH_LIST = list(MODEL_ARCH_MAP.keys())


def get_model_metadata(model_arch: str):
    """Returns the model metadata"""
    if model_arch in MODEL_ARCH_MAP:
        return MODEL_ARCH_MAP[model_arch]
    else:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")


def load_model(model_arch: str, output_dimension: int = 1, pretrained: bool = True):
    """Loads a model from a given arch and sets it up for training"""
    if model_arch not in MODEL_ARCH_MAP:
        raise NotImplementedError(f"model_arch={model_arch} is not implemented yet.")

    if MODEL_ARCH_MAP[model_arch]["library"] == "keras":
        from .keras_models import load_keras_model

        return load_keras_model(model_arch, output_dimension, pretrained)

    raise NotImplementedError(
        f"library {MODEL_ARCH_MAP[model_arch]['library']} is not implemented yet."
    )
