# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script contains methods to hangle inputs for tensorflow model training.
"""
import os
import logging
import glob
import random
import re
import tempfile

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import tensorflow


class ImageAndMaskHelper():
    def __init__(self,
                 images_dir,
                 masks_dir,
                 images_filename_pattern,
                 masks_filename_pattern):
                #  images_filename_pattern="(.*)_leftImg8bit.png",
                #  masks_filename_pattern="(.*)_gtFine_labelIds.png"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images_filename_pattern = re.compile(images_filename_pattern)
        self.masks_filename_pattern = re.compile(masks_filename_pattern)
        self.logger = logging.getLogger(__name__)

        # paths as tuples
        self.image_masks_pairs = []
        # paths as slices
        self.images = []
        self.masks = []

    def build_pair_list(self):
        parsing_stats = {
            "masks_not_matching" : 0,
            "images_not_matching" : 0,
            "images_without_masks" : 0
        }
        # search for all masks matching file name pattern
        masks_paths = []
        for file_path in glob.glob(self.masks_dir+"/**/*", recursive=True):
            matches = self.masks_filename_pattern.match(os.path.basename(file_path))
            if matches:
                masks_paths.append(
                    (
                        matches.group(1),
                        file_path
                    )
                )
            else:
                parsing_stats["masks_not_matching"] += 1
        masks_paths = dict(masks_paths) # turn list of tuples into a map

        images_paths = []
        for file_path in glob.glob(self.images_dir+"/**/*", recursive=True):
            matches = self.images_filename_pattern.match(os.path.basename(file_path))
            if matches:
                images_paths.append(
                    (
                        matches.group(1),
                        file_path
                    )
                )
            else:
                parsing_stats["images_not_matching"] += 1

        self.images = []
        self.masks = []
        self.image_masks_pairs = []
        for image_key, image_path in images_paths:
            if image_key in masks_paths:
                self.images.append(image_path)
                self.masks.append(masks_paths[image_key])
                self.image_masks_pairs.append(
                    (
                        image_path,
                        masks_paths[image_key]
                    )
                )
            else:
                self.logger.debug(f"Image {image_path} doesn't have a corresponding mask.")
                parsing_stats["images_without_masks"] += 1

        parsing_stats["found_pairs"] = len(self.image_masks_pairs)

        self.logger.info(f"Finished parsing images/masks paths: {parsing_stats}")

        return self.image_masks_pairs

    def __len__(self):
        return len(self.image_masks_pairs)


class ImageAndMaskSequenceDataset(ImageAndMaskHelper):
    @staticmethod
    def loading_function(image_path, mask_path, target_size):
        #logging.getLogger(__name__).info(f"Actually loading image {image_path}")
        image = np.array(load_img(image_path, target_size=(target_size,target_size)))

        mask = np.array(load_img(mask_path, target_size=(target_size,target_size), color_mode="grayscale"))
        mask = np.expand_dims(mask, 2)
        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        mask -= 1
        #mask = np.clip(mask, 0, self.num_classes)

        #assert image.shape == self.img_size + (3,)
        #assert mask.shape == self.img_size + (1,)

        return image, mask
    
    def dataset(self, input_size=160, training=False, num_shards=1, shard_index=0, cache=None, batch_size=64, prefetch_factor=2, prefetch_workers=5, num_classes=3):
        image_and_mask_path_list = self.build_pair_list()

        def _generator():
            """ Returns generator """
            # see https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
            for image_path, mask_path in image_and_mask_path_list:
                yield ImageAndMaskSequenceDataset.loading_function(image_path, mask_path, input_size)

        # https://cs230.stanford.edu/blog/datapipeline/#best-practices
        with tensorflow.device("CPU"):
            _dataset = tensorflow.data.Dataset.from_generator(
                _generator,
                output_signature=(
                    tensorflow.TensorSpec(shape=(input_size,input_size,3), dtype=tensorflow.int32),
                    tensorflow.TensorSpec(shape=(input_size,input_size,1), dtype=tensorflow.uint8)
                )
            )

            #self.training_dataset = self.training_dataset.shard(num_shards=self.nodes, index=self.worker_id)
            #self.validation_dataset = self.validation_dataset.shard(num_shards=self.nodes, index=self.worker_id)

            #self.training_dataset = self.training_dataset.map(load_function, num_parallel_calls=self.dataloading_config.num_workers)
            #self.validation_dataset = self.validation_dataset.map(load_function, num_parallel_calls=self.dataloading_config.num_workers)

            if cache == "disk":
                _dataset = _dataset.cache(tempfile.NamedTemporaryFile().name)
            elif cache == "memory":
                _dataset = _dataset.cache()

            #_dataset = _dataset.shuffle(len(shard_image_masks_pairs))
            _dataset = _dataset.batch(batch_size)

            if prefetch_factor < 0:
                # by default, use AUTOTUNE
                _dataset = _dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
            elif prefetch_factor > 0:
                _dataset = _dataset.prefetch(buffer_size=prefetch_factor)

            # # Disable AutoShard.
            options = tensorflow.data.Options()
            options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
            _dataset = _dataset.with_options(options)

        return _dataset
