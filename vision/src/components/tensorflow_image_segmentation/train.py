# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script implements a Distributed PyTorch training sequence.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed pytorch
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
- PROFILER : how to implement pytorch profiler
"""
import os
import sys
import time
import json
import logging
import argparse
import traceback
from tqdm import tqdm
from distutils.util import strtobool
import random

import mlflow

# the long list of tensorflow imports
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import distribute

# add path to here, if necessary
COMPONENT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), ".")
)
if COMPONENT_ROOT not in sys.path:
    logging.info(f"Adding {COMPONENT_ROOT} to path")
    sys.path.append(str(COMPONENT_ROOT))

from profiling import LogTimeBlock, LogDiskIOBlock, LogTimeOfIterator


from image_io import build_image_segmentation_datasets
from model import get_model_metadata, load_model


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group(f"Training Inputs")
    group.add_argument(
        "--images",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    group.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="path to folder containing image annotations",
    )

    group = parser.add_argument_group(f"Training Outputs")
    group.add_argument(
        "--model_output",
        type=str,
        required=False,
        default=None,
        help="Path to write final model",
    )
    group.add_argument(
        "--checkpoints",
        type=str,
        required=False,
        default=None,
        help="Path to read/write checkpoints",
    )
    group.add_argument(
        "--register_model_as",
        type=str,
        required=False,
        default=None,
        help="Name to register final model in MLFlow",
    )

    group = parser.add_argument_group(f"Data Loading Parameters")
    group.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train/valid data loading batch size (default: 64)",
    )

    group = parser.add_argument_group(f"Model/Training Parameters")
    group.add_argument(
        "--model_arch",
        type=str,
        required=False,
        default="unet",
        help="Which model architecture to use (default: unet)",
    )
    group.add_argument(
        "--model_input_size",
        type=int,
        required=False,
        default=160,
        help="Size of input images (resized)"
    )
    group.add_argument(
        "--model_arch_pretrained",
        type=strtobool,
        required=False,
        default=True,
        help="Use pretrained model (default: true)",
    )
    group.add_argument(
        "--disable_cuda",
        type=strtobool,
        required=False,
        default=False,
        help="set True to force use of cpu (local testing).",
    )
    group.add_argument(
        "--distributed_strategy",
        type=str,
        required=False,
        # see https://www.tensorflow.org/guide/distributed_training
        choices=[
            "MirroredStrategy",
            #"TPUStrategy",
            "MultiWorkerMirroredStrategy",
            #"CentralStorageStrategy",
            #"ParameterServerStrategy"
        ],
        default="MirroredStrategy",
        help="Which distributed strategy to use.",
    )
    group.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        default=1,
        help="Number of epochs to train for",
    )
    group.add_argument(
        "--optimizer",
        type=str,
        required=False,
        default="rmsprop",
    )
    group.add_argument(
        "--loss",
        type=str,
        required=False,
        default="sparse_categorical_crossentropy",
    )
    # group.add_argument(
    #     "--learning_rate",
    #     type=float,
    #     required=False,
    #     default=0.001,
    #     help="Learning rate of optimizer",
    # )

    group = parser.add_argument_group(f"System Parameters")
    # group.add_argument(
    #     "--enable_profiling",
    #     type=strtobool,
    #     required=False,
    #     default=False,
    #     help="Enable pytorch profiler.",
    # )

    return parser


def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # Get distribution config
    if "TF_CONFIG" not in os.environ:
        logger.critical("TF_CONFIG cannot be found in os.environ")

    if args.disable_cuda:
        logger.warning(f"Cuda disabled by replacing current CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} by '-1'")
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    strategy = distribute.MultiWorkerMirroredStrategy(
        #devices=None
    )

    # DATA
    train_gen, val_gen = build_image_segmentation_datasets(
        images_dir = args.images,
        annotations_dir = args.annotations,
        val_samples = 1000,
        input_size = args.model_input_size,
        batch_size = args.batch_size
    )

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    with strategy.scope():
        model = load_model(
            model_arch=args.model_arch,
            input_size=args.model_input_size,
            num_classes=3
        )

        # Configure the model for training.
        # We use the "sparse" version of categorical_crossentropy
        # because our target data is integers.
        model.compile(optimizer=args.optimizer, loss=args.loss)

    callbacks = [
        # keras.callbacks.ModelCheckpoint("segmentation.h5", save_best_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    model.fit(train_gen, epochs=args.num_epochs, validation_data=val_gen, callbacks=callbacks)

    # MLFLOW: finalize mlflow (once in entire script)
    mlflow.end_run()

    logger.info("run() completed")



def main(cli_args=None):
    """Main function of the script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
