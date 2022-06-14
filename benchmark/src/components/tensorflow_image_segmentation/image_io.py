# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for tensorflow model training.
"""
import os
import logging
import glob
import random

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import tensorflow

class ImageSegmentationSequence(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, img_size, input_img_paths, target_img_paths):
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths)

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        image = np.array(load_img(self.input_img_paths[idx], target_size=self.img_size))

        mask = np.array(load_img(self.target_img_paths[idx], target_size=self.img_size, color_mode="grayscale"))
        mask = np.expand_dims(mask, 2)
        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        mask -= 1

        assert image.shape == self.img_size + (3,)
        assert mask.shape == self.img_size + (1,)

        return image, mask

    def generator(self):
        """ Returns generator """
        # see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)


def build_image_segmentation_datasets(
    images_dir: str,
    annotations_dir: str,
    val_samples: int = 1000,
    input_size: int = 160,
    image_extension: str = "jpg",
    annotations_extension: str = "png"
):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images
        annotations_dir (str): path to the directory containing annotations
        input_size (int): input size expected by the model

    Returns:
        train_dataset (ImageSegmentationSequence): training dataset
        valid_dataset (ImageSegmentationSequence): validation dataset
    """
    logger = logging.getLogger(__name__)

    logger.info(
        f"Creating train/val datasets from {images_dir}"
    )

    input_img_paths = sorted(
        [
            os.path.join(images_dir, fname)
            for fname in os.listdir(images_dir)
            if fname.lower().endswith(image_extension)
        ]
    )
    annotations_img_paths = sorted(
        [
            os.path.join(annotations_dir, fname)
            for fname in os.listdir(annotations_dir)
            if fname.lower().endswith(annotations_extension) and not fname.startswith(".")
        ]
    )

    logger.info(
        f"{len(input_img_paths)} images found in {images_dir}"
    )
    for input_path, target_path in zip(input_img_paths[:10], annotations_img_paths[:10]):
        logger.info(f"img[]: {input_path} => {target_path}")

    # Split our img paths into a training and a validation set
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(annotations_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = annotations_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = annotations_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = ImageSegmentationSequence(
        (input_size, input_size),
        train_input_img_paths,
        train_target_img_paths
    )
    train_dataset = tensorflow.data.Dataset.from_generator(
        train_gen.generator,
        output_signature=(
            tensorflow.TensorSpec(shape=(input_size,input_size,3), dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=(input_size,input_size,1), dtype=tensorflow.uint8)
        )
    )
    val_gen = ImageSegmentationSequence(
        (input_size, input_size),
        val_input_img_paths,
        val_target_img_paths
    )
    val_dataset = tensorflow.data.Dataset.from_generator(
        val_gen.generator,
        output_signature=(
            tensorflow.TensorSpec(shape=(input_size,input_size,3), dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=(input_size,input_size,1), dtype=tensorflow.uint8)
        )
    )

    return train_dataset, val_dataset
