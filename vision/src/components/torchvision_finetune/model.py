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

MODEL_ARCH_LIST = [
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
    # "squeezenet",
    # "densenet",
    # "inception",
    # "googlenet",
    # "shufflenet",
    # "mobilenet_v2",
    # "mobilenet_v3_large",
    # "mobilenet_v3_small",
    # "resnext50_32x4d",
    # "wide_resnet50_2",
    # "mnasnet",
    # "efficientnet_b0",
    # "efficientnet_b1",
    # "efficientnet_b2",
    # "efficientnet_b3",
    # "efficientnet_b4",
    # "efficientnet_b5",
    # "efficientnet_b6",
    # "efficientnet_b7",
    # "regnet_y_400mf",
    # "regnet_y_800mf",
    # "regnet_y_1_6gf",
    # "regnet_y_3_2gf",
    # "regnet_y_8gf",
    # "regnet_y_16gf",
    # "regnet_y_32gf",
    # "regnet_x_400mf",
    # "regnet_x_800mf",
    # "regnet_x_1_6gf",
    # "regnet_x_3_2gf",
    # "regnet_x_8gf",
    # "regnet_x_16gf",
    # "regnet_x_32gf",
]

MODEL_ARCH_INPUT_SIZES = {
    "resnet18" : 224,
    "resnet34" : 224,
    "resnet50" : 224,
    "resnet101" : 224,
    "resnet152" : 224,
    "alexnet" : 224,
    "vgg11" : 224,
    "vgg11_bn" : 224,
    "vgg13" : 224,
    "vgg13_bn" : 224,
    "vgg16" : 224,
    "vgg16_bn" : 224,
    "vgg19" : 224,
    "vgg19_bn" : 224,
}


def load_and_model_arch(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if model_arch in MODEL_ARCH_LIST:
        model = getattr(models, model_arch)(pretrained=pretrained)
    else:
        raise NotImplementedError(
            f"model_arch={model_arch} is not implemented in torchvision model zoo."
        )

    # see https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if model_arch.startswith("resnet"):
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, output_dimension),
            torch.nn.Softmax(dim=1)   # adding Softmax to output probs
        )
    elif model_arch == "alexnet":
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, output_dimension),
            torch.nn.Softmax(dim=1)   # adding Softmax to output probs
        )
    elif model_arch.startswith("vgg"):
        model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, output_dimension),
            torch.nn.Softmax(dim=1)   # adding Softmax to output probs
        )
    elif model_arch == "densenet":
        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, output_dimension),
            torch.nn.Softmax(dim=1)   # adding Softmax to output probs
        )
    else:
        raise NotImplementedError(
            f"loading model_arch={model_arch} is not implemented yet in our custom code."
        )

    return model
