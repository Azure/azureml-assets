# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for pytorch model training
using the COCO dataset https://cocodataset.org/.
"""
import os
import logging
import csv
import glob

import torch
import torchvision


def build_image_datasets(
    train_images_dir: str,
    valid_images_dir: str,
):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images

    Returns:
        train_dataset (torchvision.datasets.VisionDataset): training dataset
        valid_dataset (torchvision.datasets.VisionDataset): validation dataset
        labels (Dict[str, int]): labels
    """
    logger = logging.getLogger(__name__)

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(200),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    train_dataset = torchvision.datasets.ImageFolder(
        root=train_images_dir, transform=train_transform
    )
    logger.info(
        f"ImageFolder loaded training image list samples={len(train_dataset)}, has classes {train_dataset.classes}"
    )

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(200),
            torchvision.transforms.CenterCrop(200),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.405], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    valid_dataset = torchvision.datasets.ImageFolder(
        root=valid_images_dir, transform=valid_transform
    )

    logger.info(
        f"ImageFolder loaded validation image list samples={len(valid_dataset)}, has classes {train_dataset.classes}"
    )

    return train_dataset, valid_dataset, train_dataset.classes
