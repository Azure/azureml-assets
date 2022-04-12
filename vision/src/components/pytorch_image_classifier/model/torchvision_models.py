# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

TORCHVISION_MODEL_ARCH_LIST = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "alexnet",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161"
]


def load_torchvision_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if model_arch in TORCHVISION_MODEL_ARCH_LIST:
        model = getattr(models, model_arch)(pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"model_arch={model_arch} is not implemented in torchvision model zoo."
        )

    # see https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if model_arch.startswith("resnet"):
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, output_dimension),
            torch.nn.Softmax(dim=1),  # adding Softmax to output probs
        )
    elif model_arch == "alexnet":
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, output_dimension),
            torch.nn.Softmax(dim=1),  # adding Softmax to output probs
        )
    elif model_arch.startswith("vgg"):
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, output_dimension),
            torch.nn.Softmax(dim=1),  # adding Softmax to output probs
        )
    elif model_arch.startswith("densenet"):
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, output_dimension),
            torch.nn.Softmax(dim=1),  # adding Softmax to output probs
        )
    else:
        raise NotImplementedError(
            f"loading model_arch={model_arch} is not implemented yet in our custom code."
        )

    return model
