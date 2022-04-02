# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from multiple libraries.
"""

MODEL_ARCH_MAP = {
    # torchvision models
    "resnet18" : { 'input_size': 224, 'library': 'torchvision' },
    "resnet34" : { 'input_size': 224, 'library': 'torchvision' },
    "resnet50" : { 'input_size': 224, 'library': 'torchvision' },
    "resnet101" : { 'input_size': 224, 'library': 'torchvision' },
    "resnet152" : { 'input_size': 224, 'library': 'torchvision' },
    "alexnet" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg11" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg11_bn" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg13" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg13_bn" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg16" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg16_bn" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg19" : { 'input_size': 224, 'library': 'torchvision' },
    "vgg19_bn" : { 'input_size': 224, 'library': 'torchvision' },

    # swin models (transformer)
    "swin-t-in1k" : { 'input_size': 224, 'library': 'swin' },
}

MODEL_ARCH_LIST = list(MODEL_ARCH_MAP.keys())

def get_model_metadata(model_arch: str):
    """Returns the model metadata"""
    if model_arch in MODEL_ARCH_MAP:
        return MODEL_ARCH_MAP[model_arch]
    else:
        return NotImplementedError(f"model_arch={model_arch} is not implemented yet.")

def load_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    if model_arch not in MODEL_ARCH_MAP:
        return NotImplementedError(f"model_arch={model_arch} is not implemented yet.")

    if MODEL_ARCH_MAP[model_arch]['library'] == 'torchvision':
        from .torchvision_models import load_torchvision_model
        return load_torchvision_model(model_arch, output_dimension, pretrained)
    if MODEL_ARCH_MAP[model_arch]['library'] == 'swin':
        from .swin_models import load_swin_model
        return load_swin_model(model_arch, output_dimension, pretrained)
    else:
        return NotImplementedError(f"library {MODEL_ARCH_MAP[model_arch]['library']} is not implemented yet.")
