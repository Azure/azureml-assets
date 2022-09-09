# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
This script implements a Distributed Tensorflow training sequence.

IMPORTANT: We have tagged the code with the following expressions to walk you through
the key implementation details.

Using your editor, search for those strings to get an idea of how to implement:
- DISTRIBUTED : how to implement distributed tensorflow
- MLFLOW : how to implement mlflow reporting of metrics and artifacts
"""
import os
import sys
import time
import logging
import argparse
from distutils.util import strtobool

import mlflow

# tensorflow imports
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# fix to AzureML PYTHONPATH
ROOT_FOLDER_PATH = os.path.join(os.path.dirname(__file__), "..")
if ROOT_FOLDER_PATH not in sys.path:
    print(f"Adding root folder to PYTHONPATH: {ROOT_FOLDER_PATH}")
    sys.path.append(ROOT_FOLDER_PATH)

# internal imports
## non-specific helper code
from common.profiling import LogTimeBlock, LogDiskIOBlock  # noqa : E402
from common.io import find_image_subfolder

## tensorflow generic helping code
from tensorflow_benchmark.helper.training import TensorflowDistributedModelTrainingSequence  # noqa : E402

## classification specific code
from tensorflow_benchmark.classification.model import get_model_metadata, load_model  # noqa : E402

SCRIPT_START_TIME = time.time()  # just to measure time to start


def run(args):
    """Run the script using CLI arguments.
    IMPORTANT: for the list of arguments, check build_argument_parser() function below.

    This function will demo the main steps for training PyTorch using a generic
    sequence provided as helper code.

    Args:
        args (argparse.Namespace): arguments parsed from CLI

    Returns:
        None
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # use a handler for the training sequence
    training_handler = TensorflowDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # DATA
    with LogTimeBlock("build_image_datasets", enabled=True), LogDiskIOBlock(
        "build_image_datasets", enabled=True
    ):
        model_input_size = get_model_metadata(args.model_arch)["input_size"]
        train_dataset = tfds.folder_dataset.ImageFolder(
            root_dir=find_image_subfolder(args.train_images),
            shape=(model_input_size, model_input_size, 3)
        ).as_dataset(
            shuffle_files=True
        )

        valid_dataset = tfds.folder_dataset.ImageFolder(
            root_dir=find_image_subfolder(args.valid_images),
            shape=(model_input_size, model_input_size, 3)
        )

        training_handler.setup_datasets(
            train_dataset,
            train_loading_function,
            val_dataset,
            val_loading_function,
            training_dataset_length=len(
                train_dataset_helper
            ),  # used to shuffle and repeat dataset
        )

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # DISTRIBUTED: build model
    with LogTimeBlock("load_model", enabled=True):
        with training_handler.strategy.scope():
            model = load_model(
                model_arch=args.model_arch,
                input_size=args.model_input_size,
                num_classes=args.num_classes,
            )

            # print model summary to stdout
            model.summary()

            # Configure the model for training.
            # We use the "sparse" version of categorical_crossentropy
            # because our target data is integers.
            model.compile(
                optimizer=args.optimizer,
                loss=args.loss,
                metrics=["accuracy"],
                # run_eagerly=True
            )

    # sets the model for distributed training
    training_handler.setup_model(model)

    mlflow.log_metric(
        "start_to_fit_time", time.time() - SCRIPT_START_TIME
    )

    # runs training sequence
    # NOTE: num_epochs is provided in args
    try:
        training_handler.train()  # TODO: checkpoints_dir=args.checkpoints)
    except RuntimeError as runtime_exception:  # if runtime error occurs (ex: cuda out of memory)
        # then print some runtime error report in the logs
        training_handler.runtime_error_report(runtime_exception)
        # re-raise
        raise runtime_exception

    # saves final model
    if args.model_output:
        training_handler.save(
            args.model_output,
            name=f"epoch-{args.num_epochs}",
            register_as=args.register_model_as,
        )

    # properly teardown distributed resources
    training_handler.close()

    # logging total time
    mlflow.log_metric("wall_time", time.time() - SCRIPT_START_TIME)

    # MLFLOW: finalize mlflow (once in entire script)
    mlflow.end_run()

    logger.info("run() completed")


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group("Training Inputs")
    group.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    group.add_argument(
        "--valid_images",
        type=str,
        required=True,
        help="path to folder containing validation images",
    )

    group = parser.add_argument_group("Training Outputs")
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

    group = parser.add_argument_group("Data Loading Parameters")
    group.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=64,
        help="Train/valid data loading batch size (default: 64)",
    )
    group.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=-1,
        help="Num workers for data loader (default: AUTOTUNE)",
    )
    group.add_argument(
        "--prefetch_factor",
        type=int,
        required=False,
        default=-1,
        help="Data loader prefetch factor (default: AUTOTUNE)",
    )
    group.add_argument(
        "--cache",
        type=str,
        required=False,
        choices=["none", "disk", "memory"],
        default="none",
        help="Use cache either on DISK or in MEMORY, or NONE",
    )

    group = parser.add_argument_group("Model/Training Parameters")
    group.add_argument(
        "--model_arch",
        type=str,
        required=False,
        default="unet",
        help="Which model architecture to use (default: unet)",
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

    group = parser.add_argument_group("Training Backend Parameters")
    group.add_argument(
        "--enable_profiling",
        type=strtobool,
        required=False,
        default=False,
        help="Enable tensorflow profiler.",
    )
    group.add_argument(
        "--disable_cuda",
        type=strtobool,
        required=False,
        default=False,
        help="set True to force use of cpu (local testing).",
    )
    group.add_argument(
        "--num_gpus",
        type=int,
        required=False,
        default=-1,
        help="limit the number of gpus to use (default: -1 for no limit).",
    )
    group.add_argument(
        "--distributed_strategy",
        type=str,
        required=False,
        # see https://www.tensorflow.org/guide/distributed_training
        choices=[
            "auto",
            "multiworkermirroredstrategy",
            "mirroredstrategy",
            "onedevicestrategy",
            "horovod",
        ],
        default="auto",  # will auto identify
        help="Which distributed strategy to use.",
    )
    group.add_argument(
        "--distributed_backend",
        type=str,
        required=False,
        choices=[
            "auto",
            "ring",
            "nccl",
        ],
        default="Auto",  # will auto identify
        help="Which backend (ring, nccl, auto) for MultiWorkerMirroredStrategy collective communication.",
    )

    return parser


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

    tf.get_logger().setLevel('INFO')

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # correct type for strtobool
    args.enable_profiling = bool(args.enable_profiling)
    args.disable_cuda = bool(args.disable_cuda)

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
