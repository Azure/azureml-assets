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
import json
import logging
import argparse
import traceback
from tqdm import tqdm
from distutils.util import strtobool
import random

import mlflow
import numpy as np

# the long list of tensorflow imports
import tensorflow
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
from profiling import CustomCallbacks


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
        "--masks",
        type=str,
        required=True,
        help="path to folder containing segmentation masks",
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
    group.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=-1,
        help="Num workers for data loader (default: -1 => all cpus available)",
    )
    group.add_argument(
        "--prefetch_factor",
        type=int,
        required=False,
        default=-1,
        help="Data loader prefetch factor (default: AUTOTUNE)",
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
        help="Size of input images (resized, default: 160)"
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
        "--num_gpus",
        type=int,
        required=False,
        default=None,
        help="limit the number of gpus to use (default: None).",
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


class TensorflowDistributedModelTrainingSequence:
    """Generic class to run the sequence for training a Tensorflow model
    using distributed training."""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(__name__)

        # DATA
        self.training_dataset = None
        self.validation_dataset = None

        # MODEL
        self.model = None

        # DISTRIBUTED CONFIG
        self.strategy = None
        self.nodes = None
        self.devices = []
        self.gpus = None
        self.distributed_available = False
        self.self_is_main_node = True
        self.cpu_count = os.cpu_count()

        # TRAINING CONFIGS
        self.dataloading_config = None
        self.training_config = None

    def setup_config(self, args):
        """Sets internal variables using provided CLI arguments (see build_arguments_parser()).
        In particular, sets device(cuda) and multinode parameters."""
        self.dataloading_config = args
        self.training_config = args

        # verify parameter default values
        if self.dataloading_config.num_workers is None:
            self.dataloading_config.num_workers = 0
        if self.dataloading_config.num_workers < 0:
            self.dataloading_config.num_workers = self.cpu_count
        if self.dataloading_config.num_workers == 0:
            self.logger.warning("You specified num_workers=0, forcing prefetch_factor to be discarded.")
            self.dataloading_config.prefetch_factor = 0

        # Get distribution config
        if "TF_CONFIG" not in os.environ:
            self.logger.critical("TF_CONFIG cannot be found in os.environ, defaulting back to non-distributed training")
            self.nodes = 1
            self.devices = [ device.name for device in tensorflow.config.list_physical_devices('GPU') ]
            self.gpus = len(self.devices)
            self.worker_id = 0
        else:
            tf_config = json.loads(os.environ['TF_CONFIG'])
            self.logger.info(f"Found TF_CONFIG = {tf_config}")
            self.nodes = len(tf_config['cluster']['worker'])
            self.devices = [ device.name for device in tensorflow.config.list_physical_devices('GPU') ]
            self.gpus = len(self.devices)
            self.worker_id = tf_config['task']['index']

        self.distributed_available = ((self.nodes * self.gpus) > 1)
        self.self_is_main_node = (self.worker_id == 0)

        self.logger.info(f"Distribution settings: nodes={self.nodes}, gpus={self.gpus}, distributed_available={self.distributed_available}, self_is_main_node={self.self_is_main_node}")

        if args.disable_cuda:
            self.logger.warning(f"Cuda disabled by replacing current CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} by '-1'")
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            self.gpus = 0
            self.devices = []
        elif args.num_gpus:
            # TODO: force down the number of gpus
            pass

        # Identify strategy
        if self.nodes > 1:
            # multiple nodes
            self.strategy = distribute.MultiWorkerMirroredStrategy()
        elif self.gpus > 1:
            # single node, multi gpu
            self.strategy = distribute.MirroredStrategy()
        else:
            # single node, single gpu
            self.strategy = distribute.OneDeviceStrategy(device="GPU:0")

        # DISTRIBUTED: in distributed mode, you want to report parameters
        # only from main process (rank==0) to avoid conflict
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            logged_params = {
                # log some distribution params
                "nodes": self.nodes,
                "gpus": self.gpus,
                #"cuda_available": not(args.disable_cuda),
                "disable_cuda": self.training_config.disable_cuda,
                "distributed": self.distributed_available,
                "distributed_strategy": "none" if self.strategy is None else self.strategy.__class__.__name__,

                # data loading params
                "batch_size": self.dataloading_config.batch_size,
                "num_workers": self.dataloading_config.num_workers,
                "cpu_count": self.cpu_count,
                "prefetch_factor": self.dataloading_config.prefetch_factor,

                # training params
                "model_arch": self.training_config.model_arch,
                "model_input_size": self.training_config.model_input_size,
            }

            if not self.training_config.disable_cuda:
                # add some gpu properties
                logged_params['cuda_device_count'] = len(tensorflow.config.list_physical_devices('GPU'))

            mlflow.log_params(logged_params)

    def setup_datasets(
        self,
        training_dataset: tensorflow.data.Dataset,
        validation_dataset: tensorflow.data.Dataset
    ):
        """Creates and sets up dataloaders for training/validation datasets."""
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset

        self.training_dataset = self.training_dataset.shard(num_shards=self.nodes, index=self.worker_id)
        self.validation_dataset = self.validation_dataset.shard(num_shards=self.nodes, index=self.worker_id)

        self.training_dataset = self.training_dataset.batch(self.dataloading_config.batch_size)
        self.validation_dataset = self.validation_dataset.batch(self.dataloading_config.batch_size)

        if self.dataloading_config.prefetch_factor < 0:
            # by default, use AUTOTUNE
            self.training_dataset = self.training_dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
            self.validation_dataset = self.validation_dataset.prefetch(buffer_size=tensorflow.data.AUTOTUNE)
        elif self.dataloading_config.prefetch_factor > 0:
            self.training_dataset = self.training_dataset.prefetch(buffer_size=self.dataloading_config.prefetch_factor)
            self.validation_dataset = self.validation_dataset.prefetch(buffer_size=self.dataloading_config.prefetch_factor)

        # # Disable AutoShard.
        options = tensorflow.data.Options()
        options.experimental_distribute.auto_shard_policy = tensorflow.data.experimental.AutoShardPolicy.OFF
        self.training_dataset = self.training_dataset.with_options(options)
        self.validation_dataset = self.validation_dataset.with_options(options)

        # Distribute input using the `experimental_distribute_dataset`.
        # training_dataset = strategy.experimental_distribute_dataset(training_dataset)
        # validation_dataset = strategy.experimental_distribute_dataset(validation_dataset)

    def setup_model(self, model):
        """Configures a model for training."""
        # Nothing specific to do here
        # for DISTRIBUTED, the model should be wrapped in strategy.scope() during model building
        self.model = model

        params_count = np.sum([np.prod(v.get_shape()) for v in self.model.trainable_weights])
        self.logger.info("MLFLOW: model_param_count={:.2f} (millions)".format(round(params_count/1e6, 2)))
        if self.self_is_main_node:
            mlflow.log_params({"model_param_count": round(params_count/1e6, 2)})

        return self.model

    def train(self, epochs:int=None, checkpoints_dir:str=None):
        """Trains the model.

        Args:
            epochs (int, optional): if not provided uses internal config
            checkpoints_dir (str, optional): path to write checkpoints
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        callbacks = [
            CustomCallbacks()
            # keras.callbacks.ModelCheckpoint("segmentation.h5", save_best_only=True)
        ]

        # Train the model, doing validation at the end of each epoch.
        self.model.fit(
            self.training_dataset,
            epochs=epochs,
            validation_data=self.validation_dataset,
            callbacks=callbacks
        )

    def runtime_error_report(self, runtime_exception):
        """Call this when catching a critical exception.
        Will print all sorts of relevant information to the log."""
        self.logger.critical(traceback.format_exc())
        try:
            import psutil
            self.logger.critical(f"Memory: {str(psutil.virtual_memory())}")
        except ModuleNotFoundError:
            self.logger.critical("import psutil failed, cannot display virtual memory stats.")

    def close(self):
        """Tear down potential resources"""
        pass # AFAIK nothing to do here

    def save(self, output_dir: str, name: str = "dev", register_as: str = None) -> None:
        # DISTRIBUTED: you want to save the model only from the main node/process
        # in data distributed mode, all models should theoretically be the same
        pass

def run(args):
    """Run the script using CLI arguments"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    # use a handler for the training sequence
    training_handler = TensorflowDistributedModelTrainingSequence()

    # sets cuda and distributed config
    training_handler.setup_config(args)

    # DATA
    with LogTimeBlock("build_image_datasets", enabled=True), LogDiskIOBlock("build_image_datasets", enabled=True):
        train_dataset, val_dataset = build_image_segmentation_datasets(
            images_dir = args.images,
            annotations_dir = args.masks,
            val_samples = 1000,
            input_size = args.model_input_size
        )

        training_handler.setup_datasets(train_dataset, val_dataset)

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # DISTRIBUTED: build model
    with LogTimeBlock("load_model", enabled=True):
        with training_handler.strategy.scope():
            model = load_model(
                model_arch=args.model_arch,
                input_size=args.model_input_size,
                num_classes=3
            )

            # Configure the model for training.
            # We use the "sparse" version of categorical_crossentropy
            # because our target data is integers.
            model.compile(
                optimizer=args.optimizer,
                loss=args.loss,
                metrics=['accuracy']
                # run_eagerly=True
            )

    # sets the model for distributed training
    training_handler.setup_model(model)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    try:
        training_handler.train() # TODO: checkpoints_dir=args.checkpoints)
    except RuntimeError as runtime_exception: # if runtime error occurs (ex: cuda out of memory)
        # then print some runtime error report in the logs
        training_handler.runtime_error_report(runtime_exception)
        # re-raise
        raise runtime_exception

    # properly teardown distributed resources
    training_handler.close()

    # saves final model
    if args.model_output:
        training_handler.save(
            args.model_output,
            name=f"epoch-{args.num_epochs}",
            register_as=args.register_model_as,
        )

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
