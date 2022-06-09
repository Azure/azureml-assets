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

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y

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
    batch_size: int = 64,
    image_extension: str = ".jpg",
    annotations_extension: str = ".png"
):
    """
    Args:
        train_images_dir (str): path to the directory containing training images
        valid_images_dir (str): path to the directory containing validation images
        annotations_dir (str): path to the directory containing annotations
        input_size (int): input size expected by the model
        batch_size (int): batch size

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
        batch_size,
        (input_size, input_size),
        train_input_img_paths,
        train_target_img_paths
    )
    train_dataset = tensorflow.data.Dataset.from_generator(
        train_gen.generator,
        output_signature=(
            tensorflow.TensorSpec(shape=(batch_size,input_size,input_size,3), dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=(batch_size,input_size,input_size,1), dtype=tensorflow.uint8)
        )
    )
    val_gen = ImageSegmentationSequence(
        batch_size,
        (input_size, input_size),
        val_input_img_paths,
        val_target_img_paths
    )
    val_dataset = tensorflow.data.Dataset.from_generator(
        val_gen.generator,
        output_signature=(
            tensorflow.TensorSpec(shape=(batch_size,input_size,input_size,3), dtype=tensorflow.int32),
            tensorflow.TensorSpec(shape=(batch_size,input_size,input_size,1), dtype=tensorflow.uint8)
        )
    )

    return train_dataset, val_dataset
