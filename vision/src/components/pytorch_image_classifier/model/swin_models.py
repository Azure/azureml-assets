# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script provides code to load and setup a variety of models from torchvision.models.
"""
import logging
import torch
from transformers import SwinForImageClassification

SWIN_MODEL_ARCH_LIST = [
    "microsoft/swin-tiny-patch4-window7-224",
    "microsoft/swin-large-patch4-window12-384-in22k",
    "microsoft/swin-base-patch4-window12-384",
    "microsoft/swin-base-patch4-window12-384-in22k",
    "microsoft/swin-large-patch4-window12-384",
    "microsoft/swin-large-patch4-window7-224-in22k",
    "microsoft/swin-base-patch4-window7-224",
    "microsoft/swin-base-patch4-window7-224-in22k",
    "microsoft/swin-small-patch4-window7-224",
    "microsoft/swin-large-patch4-window7-224",
]

def load_swin_model(
    model_arch: str, output_dimension: int = 1, pretrained: bool = True
):
    """Loads a model from a given arch and sets it up for training"""
    logger = logging.getLogger(__name__)

    logger.info(
        f"Loading model from arch={model_arch} pretrained={pretrained} output_dimension={output_dimension}"
    )
    if model_arch in SWIN_MODEL_ARCH_LIST:
        # TODO: not pretrained
        model = SwinForImageClassification.from_pretrained(model_arch)

        model.classifier = torch.nn.Sequential(
            torch.nn.Linear(model.swin.num_features, output_dimension),
            torch.nn.Softmax(dim=1)   # adding Softmax to output probs
        )
    else:
        raise NotImplementedError(
            f"model_arch={model_arch} is not implemented."
        )

    return model
