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
import tempfile
import math

import mlflow
import numpy as np

# tensorflow imports
import tensorflow as tf
from tensorflow import keras

# add path to here, if necessary
COMPONENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if COMPONENT_ROOT not in sys.path:
    logging.info(f"Adding {COMPONENT_ROOT} to path")
    sys.path.append(str(COMPONENT_ROOT))

from tf_helper.profiling import LogTimeBlock, LogDiskIOBlock, LogTimeOfIterator
from tf_helper.profiling import CustomCallbacks

from tf_helper.image_io import ImageAndMaskSequenceDataset
from tf_helper.model import get_model_metadata, load_model

SCRIPT_START_TIME = time.time()


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Builds the argument parser for CLI settings"""
    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group(f"Training Inputs")
    group.add_argument(
        "--train_images",
        type=str,
        required=True,
        help="Path to folder containing training images",
    )
    group.add_argument(
        "--images_filename_pattern",
        type=str,
        required=False,
        default="(.*)\\.jpg",
        help="Regex used to find and match images with masks (matched on group(1))",
    )
    group.add_argument(
        "--images_type",
        type=str,
        required=False,
        choices=["png", "jpg"],
        default="png",
        help="png (default) or jpg",
    )
    group.add_argument(
        "--train_masks",
        type=str,
        required=True,
        help="path to folder containing segmentation masks",
    )
    group.add_argument(
        "--masks_filename_pattern",
        type=str,
        required=False,
        default="(.*)\\.png",
        help="Regex used to find and match images with masks (matched on group(1))",
    )
    group.add_argument(
        "--test_images",
        type=str,
        required=True,
        help="Path to folder containing testing images",
    )
    group.add_argument(
        "--test_masks",
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
        choices=["disk", "memory"],
        default=None,
        help="Use cache either on DISK or in MEMORY",
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
        help="Size of input images (resized, default: 160)",
    )
    group.add_argument(
        "--num_classes", type=int, required=True, help="Number of classes"
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

    group = parser.add_argument_group(f"Training Backend Parameters")
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
        default=None,
        help="limit the number of gpus to use (default: None).",
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


class TensorflowDistributedModelTrainingSequence:
    """Generic class to run the sequence for training a Tensorflow model
    using distributed training."""

    def __init__(self):
        """Constructor"""
        self.logger = logging.getLogger(__name__)

        # DATA
        self.training_dataset = None
        self.training_dataset_length = None
        self.training_steps_per_epoch = None
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
            self.dataloading_config.num_workers = tf.data.AUTOTUNE
        if self.dataloading_config.num_workers < 0:
            self.dataloading_config.num_workers = tf.data.AUTOTUNE
        if self.dataloading_config.num_workers == 0:
            self.logger.warning(
                "You specified num_workers=0, forcing prefetch_factor to be discarded."
            )
            self.dataloading_config.prefetch_factor = 0

        # Get distribution config
        if "TF_CONFIG" not in os.environ:
            self.logger.critical(
                "TF_CONFIG cannot be found in os.environ, defaulting back to non-distributed training"
            )
            self.nodes = 1
            # self.devices = [ device.name for device in tf.config.list_physical_devices('GPU') ]
            self.worker_id = 0
        else:
            tf_config = json.loads(os.environ["TF_CONFIG"])
            self.logger.info(f"Found TF_CONFIG = {tf_config}")
            self.nodes = len(tf_config["cluster"]["worker"])
            # self.devices = [ device.name for device in tf.config.list_physical_devices('GPU') ]
            self.worker_id = tf_config["task"]["index"]

        # Reduce number of GPUs artificially if requested
        if args.disable_cuda:
            self.logger.warning(f"CUDA disabled because --disable_cuda True")
            self.gpus = 0
        elif args.num_gpus == 0:
            self.logger.warning(f"CUDA disabled because --num_gpus=0")
            self.gpus = 0
        elif args.num_gpus and args.num_gpus > 0:
            # TODO: force down the number of gpus
            self.gpus = args.num_gpus
            self.logger.warning(
                f"Because you set --num_gpus={args.num_gpus}, retricting to first {self.gpus} physical devices"
            )
        else:
            self.gpus = len(tf.config.list_physical_devices("GPU"))

        # Check if we need distributed at all
        self.distributed_available = (self.nodes > 1) or (
            (self.nodes * self.gpus) > 1
        )  # if multi-node (CPU or GPU) or multi-gpu
        self.self_is_main_node = self.worker_id == 0
        self.logger.info(
            f"Distribution settings: nodes={self.nodes}, gpus={self.gpus}, distributed_available={self.distributed_available}, self_is_main_node={self.self_is_main_node}"
        )

        # Setting up TF distributed is a whole story
        self._setup_distribution_strategy()

        # DISTRIBUTED: in distributed mode, you want to report parameters
        # only from main process (rank==0) to avoid conflict
        if self.self_is_main_node:
            # MLFLOW: report relevant parameters using mlflow
            logged_params = {
                # log some distribution params
                "nodes": self.nodes,
                "instance_per_node": self.gpus,
                "disable_cuda": self.training_config.disable_cuda,
                "distributed": self.distributed_available,
                "distributed_strategy": self.training_config.distributed_strategy,
                "distributed_backend": self.training_config.distributed_backend,
                # data loading params
                "batch_size": self.dataloading_config.batch_size,
                "num_workers": self.dataloading_config.num_workers,
                "cpu_count": self.cpu_count,
                "prefetch_factor": self.dataloading_config.prefetch_factor,
                "cache": self.dataloading_config.cache,
                # training params
                "model_arch": self.training_config.model_arch,
                "model_input_size": self.training_config.model_input_size,
                "model_arch_pretrained": False,  # TODO
                "num_classes": self.training_config.num_classes,
                # profiling
                "enable_profiling": False,  # TODO
            }

            if not self.training_config.disable_cuda:
                # add some gpu properties
                logged_params["cuda_device_count"] = len(
                    tf.config.list_physical_devices("GPU")
                )
                logged_params["cuda_available"] = logged_params["cuda_device_count"] > 0

            mlflow.log_params(logged_params)

    def _setup_distribution_strategy(self):
        """DISTRIBUTED: this takes care of initializing the distribution strategy.

        Tensorflow uses a different "strategy" for each use case:
        - multi-node => MultiWorkerMirroredStrategy
        - single-node multi-gpu => MirroredStrategy
        - single-node single-gpu => OneDeviceStrategy

        Each comes with its own initialization process."""
        # Identify the right strategy depending on params + context
        if self.training_config.distributed_strategy == "auto":
            # Auto detect
            if self.nodes > 1:  # MULTI-NODE
                self.training_config.distributed_strategy = (
                    "multiworkermirroredstrategy"
                )
            elif self.gpus > 1:  # SINGLE-NODE MULTI-GPU
                self.training_config.distributed_strategy = "mirroredstrategy"
            else:  # SINGLE-NODE SINGLE-GPU
                self.training_config.distributed_strategy = "onedevicestrategy"

        if self.training_config.distributed_strategy == "multiworkermirroredstrategy":
            self.logger.info(
                "Using MultiWorkerMirroredStrategy as distributed_strategy"
            )

            # first we need to define the communication options (backend)
            if self.training_config.distributed_backend.lower() == "nccl":
                self.logger.info(
                    "Setting CommunicationImplementation.NCCL as distributed_backend"
                )
                communication_options = tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
                )
            elif self.training_config.distributed_backend.lower() == "ring":
                self.logger.info(
                    "Setting CommunicationImplementation.RING as distributed_backend"
                )
                communication_options = tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CommunicationImplementation.RING
                )
            else:
                self.logger.info(
                    "Setting CommunicationImplementation.AUTO as distributed_backend"
                )
                communication_options = tf.distribute.experimental.CommunicationOptions(
                    implementation=tf.distribute.experimental.CommunicationImplementation.AUTO
                )

            # second, we can artificially limit the number of gpus by using tf.config.set_visible_devices()
            self.devices = tf.config.list_physical_devices("GPU")[
                : self.gpus
            ]  # artificially limit visible GPU
            self.devices += tf.config.list_physical_devices("CPU")  # but add all CPU
            self.logger.info(
                f"Setting tf.config.set_visible_devices(devices={self.devices})"
            )
            tf.config.set_visible_devices(devices=self.devices)

            # finally we can initialize the strategy
            self.logger.info(f"Initialize MultiWorkerMirroredStrategy()...")
            self.strategy = tf.distribute.MultiWorkerMirroredStrategy(
                communication_options=communication_options
            )

            # we're storing the name of the strategy to log as a parameter
            self.training_config.distributed_strategy = self.strategy.__class__.__name__

        elif self.training_config.distributed_strategy == "mirroredstrategy":
            self.devices = [
                f"GPU:{i}" for i in range(self.gpus)
            ]  # artificially limit number of gpus (if requested)
            self.logger.info(
                f"Using MirroredStrategy(devices={self.devices}) as distributed_strategy"
            )

            # names of devices for MirroredStrategy must be GPU:N
            self.strategy = tf.distribute.MirroredStrategy(devices=self.devices)
            self.training_config.distributed_strategy = self.strategy.__class__.__name__

        elif self.training_config.distributed_strategy == "onedevicestrategy":
            self.devices = [f"GPU:{i}" for i in range(self.gpus)]
            self.logger.info(
                f"Using OneDeviceStrategy(devices=GPU:0) as distributed_strategy"
            )
            self.strategy = tf.distribute.OneDeviceStrategy(device="GPU:0")
            self.training_config.distributed_strategy = self.strategy.__class__.__name__

        else:
            raise ValueError(
                f"distributed_strategy={self.training_config.distributed_strategy} is not recognized."
            )

    def setup_datasets(
        self,
        training_dataset: tf.data.Dataset,
        training_dataset_loading_function: callable,
        validation_dataset: tf.data.Dataset,
        validation_dataset_loading_function: callable,
        training_dataset_length,
    ):
        """Creates the data loading pipelines training/validation datasets."""
        # let's log a couple more params
        data_params = {}

        ### 1. Initialize TRAINING dataset ###
        data_params["train_dataset_length"] = training_dataset_length
        data_params["train_dataset_shard_length"] = math.ceil(training_dataset_length / self.nodes)

        # shard(): this node will have a unique subset of data
        _dataset = training_dataset.shard(num_shards=self.nodes, index=self.worker_id)

        # shuffle(): create a random order
        _dataset = _dataset.shuffle(data_params["train_dataset_shard_length"])

        # map(): actually load the data using loading function
        _dataset = _dataset.map(
            training_dataset_loading_function,
            num_parallel_calls=self.training_config.num_workers,
        )

        # batch(): create batches
        _dataset = _dataset.batch(self.training_config.batch_size)

        # cache if you ask nicely
        if self.training_config.cache == "memory":
            _dataset = _dataset.cache()

        # repeat(): create an infinitely long dataset, required to use experimental_distribute_dataset()
        # will require steps_per_epoch argument in model.fit()
        _dataset = _dataset.repeat()

        # disable auto-sharding (since we use shard() ourselves)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )  # Disable AutoShard.
        _dataset = _dataset.with_options(options)

        # store internally as training_dataset
        self.training_dataset = _dataset
        self.training_dataset_length = training_dataset_length
        self.training_steps_per_epoch = math.ceil(
            # counts how many batches in this shard
            data_params["train_dataset_shard_length"] / self.training_config.batch_size
        )
        data_params["steps_per_epoch"] = self.training_steps_per_epoch
        self.logger.info(
            f"Training dataset is set (len={self.training_dataset_length}, batch_size{self.training_config.batch_size}, steps_per_epoch={self.training_steps_per_epoch})"
        )

        ### 2. Initialize VALIDATION dataset ###
        # NOTE: we do not use shuffle() or repeat() here, they are not necessary
        _dataset = validation_dataset.shard(num_shards=self.nodes, index=self.worker_id)
        _dataset = _dataset.map(
            validation_dataset_loading_function,
            num_parallel_calls=self.training_config.num_workers,
        )
        _dataset = _dataset.batch(self.training_config.batch_size)

        if self.training_config.cache == "memory":
            _dataset = _dataset.cache()

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF
        )  # Disable AutoShard.
        _dataset = _dataset.with_options(options)
        self.validation_dataset = _dataset
        self.logger.info(
            f"Validation dataset is set (batch_size{self.training_config.batch_size})"
        )

        ### 3. Set the TRAINING dataset for distributed training using experimental_distribute_dataset
        # see https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy#experimental_distribute_dataset
        if (
            self.training_config.distributed_strategy == "MirroredStrategy"
            or self.training_config.distributed_strategy == "MultiWorkerMirroredStrategy"
        ):
            self.logger.info(
                f"Using {self.training_config.distributed_strategy}.experimental_distribute_dataset()"
            )

            # see https://www.tensorflow.org/api_docs/python/tf/distribute/InputOptions
            input_options = tf.distribute.InputOptions(
                experimental_fetch_to_device=True,
                experimental_replication_mode=tf.distribute.InputReplicationMode.PER_WORKER,
                experimental_place_dataset_on_device=False,
                experimental_per_replica_buffer_size=self.training_config.prefetch_factor,
            )

            self.training_dataset = self.strategy.experimental_distribute_dataset(
                self.training_dataset, options=input_options
            )

        if self.self_is_main_node:
            mlflow.log_params(data_params)

    def setup_model(self, model):
        """Configures a model for training."""
        # Nothing specific to do here
        # for DISTRIBUTED, the model should be wrapped in strategy.scope() during model building
        self.model = model

        params_count = np.sum(
            [np.prod(v.get_shape()) for v in self.model.trainable_weights]
        )
        self.logger.info(
            "MLFLOW: model_param_count={:.2f} (millions)".format(
                round(params_count / 1e6, 2)
            )
        )
        if self.self_is_main_node:
            mlflow.log_params({"model_param_count": round(params_count / 1e6, 2)})

        return self.model

    def train(self, epochs: int = None, checkpoints_dir: str = None):
        """Trains the model.

        Args:
            epochs (int, optional): if not provided uses internal config
            checkpoints_dir (str, optional): path to write checkpoints
        """
        if epochs is None:
            epochs = self.training_config.num_epochs

        custom_callback_handler = CustomCallbacks(
            enabled=self.self_is_main_node, script_start_time=SCRIPT_START_TIME
        )

        callbacks = [
            custom_callback_handler,
            # keras.callbacks.ModelCheckpoint("segmentation.h5", save_best_only=True)
        ]

        custom_callback_handler.log_start_fit()

        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)

        # Train the model, doing validation at the end of each epoch.
        self.model.fit(
            self.training_dataset,
            epochs=epochs,
            steps_per_epoch=self.training_steps_per_epoch,
            validation_data=self.validation_dataset,
            callbacks=callbacks,
        )

    def runtime_error_report(self, runtime_exception):
        """Call this when catching a critical exception.
        Will print all sorts of relevant information to the log."""
        self.logger.critical(traceback.format_exc())
        try:
            import psutil

            self.logger.critical(f"Memory: {str(psutil.virtual_memory())}")
        except ModuleNotFoundError:
            self.logger.critical(
                "import psutil failed, cannot display virtual memory stats."
            )

    def close(self):
        """Tear down potential resources"""
        pass  # AFAIK nothing to do here

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
    with LogTimeBlock("build_image_datasets", enabled=True), LogDiskIOBlock(
        "build_image_datasets", enabled=True
    ):
        train_dataset_helper = ImageAndMaskSequenceDataset(
            images_dir=args.train_images,
            masks_dir=args.train_masks,
            images_filename_pattern=args.images_filename_pattern,
            masks_filename_pattern=args.masks_filename_pattern,
            images_type=args.images_type,
        )

        # the helper returns a dataset containing paths to images
        # and a loading function to use map() for loading the data
        train_dataset, train_loading_function = train_dataset_helper.dataset(
            num_classes=args.num_classes+1,
            input_size=args.model_input_size
        )

        test_dataset_helper = ImageAndMaskSequenceDataset(
            images_dir=args.test_images,
            masks_dir=args.test_masks,
            images_filename_pattern=args.images_filename_pattern,
            masks_filename_pattern=args.masks_filename_pattern,
            images_type="png",  # masks need to be in png
        )

        # the helper returns a dataset containing paths to images
        # and a loading function to use map() for loading the data
        val_dataset, val_loading_function = test_dataset_helper.dataset(
            num_classes=args.num_classes+1,
            input_size=args.model_input_size
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
                num_classes=(args.num_classes + 1), # n+1 as undefined class
            )

            # print model summary to stdout
            model.summary()

            # Configure the model for training.
            # We use the "sparse" version of categorical_crossentropy
            # because our target data is integers.

            if args.loss == "sparse_softmax_cross_entropy_with_logits":
                def _custom_loss(target, output):
                    target = tf.reshape(target, [-1])
                    output_shape = tf.shape(output)
                    output = tf.reshape(output, [output_shape[-1], -1])
                    valid_mask = tf.math.less_equal(target, args.num_classes)
                    target = target[valid_mask]
                    output = output[valid_mask]
                    return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)
                loss_func = _custom_loss
            else:
                loss_func = args.loss # use name

            model.compile(
                optimizer=args.optimizer,
                loss=loss_func,
                metrics=["accuracy"], # tf.keras.metrics.MeanIoU(num_classes=(args.num_classes+1))],
                # run_eagerly=True
                #options = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom=True)
            )

    # sets the model for distributed training
    training_handler.setup_model(model)

    # runs training sequence
    # NOTE: num_epochs is provided in args
    try:
        training_handler.train()  # TODO: checkpoints_dir=args.checkpoints)
    except RuntimeError as runtime_exception:  # if runtime error occurs (ex: cuda out of memory)
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

    # logging total time
    mlflow.log_metric("wall_time", time.time() - SCRIPT_START_TIME)

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
