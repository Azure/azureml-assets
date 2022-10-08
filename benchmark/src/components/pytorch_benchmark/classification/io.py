# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This script contains methods to handle building ImageFolder for image classification.
"""
import logging
import torchvision
from common.io import find_image_subfolder


def build_image_datasets(
    train_images_dir: str,
    valid_images_dir: str,
    input_size: int = 224,
):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images
        input_size (int): input size expected by the model

    Returns:
        train_dataset (torchvision.datasets.VisionDataset): training dataset
        valid_dataset (torchvision.datasets.VisionDataset): validation dataset
        labels (Dict[str, int]): labels
    """
    logger = logging.getLogger(__name__)

    # identify the right level of sub directory
    train_images_dir = find_image_subfolder(train_images_dir)

    logger.info(f"Creating training dataset from {train_images_dir}")

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(input_size),
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
        f"ImageFolder loaded training image from {train_images_dir}: samples={len(train_dataset)}, #classes={len(train_dataset.classes)} classes={train_dataset.classes}"
    )

    # identify the right level of sub directory
    valid_images_dir = find_image_subfolder(valid_images_dir)

    logger.info(f"Creating validation dataset from {valid_images_dir}")

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(input_size),
            torchvision.transforms.CenterCrop(input_size),
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
        f"ImageFolder loaded validation image from {valid_images_dir}: samples={len(valid_dataset)}, #classes={len(valid_dataset.classes)} classes={valid_dataset.classes}"
    )

    return train_dataset, valid_dataset, train_dataset.classes
