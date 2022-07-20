# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import tensorflow as tf


def load_keras_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )

    if model_arch.startswith("resnet50"):
        model = tf.keras.applications.resnet50.ResNet50(
            include_top=False,
            weights='imagenet' if pretrained else None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=output_dimension
        )

    else:
        raise NotImplementedError(
            f"loading model_arch={model_arch} is not implemented yet in our custom code."
        )

    return model
