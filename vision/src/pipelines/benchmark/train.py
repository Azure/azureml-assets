"""
Azure ML Pipeline Script - Runs the torchvision image classification
training component on a given training/validation dataset.
"""
import os
import sys
import argparse
from typing import List, Any

# handle to the workspace
from azure.ml import MLClient

# authentication package
from azure.identity import AzureCliCredential

# python sdk
from azure.ml import dsl

# add path to here, if necessary
PIPELINES_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PIPELINES_ROOT not in sys.path:
    print(f"Adding {PIPELINES_ROOT} to path")
    sys.path.append(str(PIPELINES_ROOT))

from common.main import main, VISION_COMPONENTS_FOLDER


def build_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Constructs the argument parser"""
    parser.add_argument(
        "--training_dataset",
        required=False,
        type=str,
        default="stanford_dogs:1",
        help="NAME:VERSION of training dataset (default: stanford_dogs:1)",
    )
    parser.add_argument(
        "--validation_dataset",
        required=False,
        type=str,
        default="stanford_dogs:1",
        help="NAME:VERSION of training dataset (default: stanford_dogs:1)",
    )
    parser.add_argument(
        "--epochs",
        required=False,
        type=int,
        default=5,
        help="number of epochs (default: 5)",
    )
    # parser.add_argument(
    #     "--input_mode",
    #     required=False,
    #     type=str,
    #     choices=['rw_mount', 'ro_mount', 'download'],
    #     default="ro_mount",
    #     help="which mode to use for input data",
    # )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        required=False,
        default=2,
        help="ata loader prefetch factor (default: 2)",
    )
    parser.add_argument(
        "--num_workers",
        required=False,
        type=int,
        default=-1,
        help="number of workers for pre-fetching (default: -1 to use all available cpus)",
    )
    parser.add_argument(
        "--enable_profiling",
        required=False,
        action="store_true",
        default=False,
        help="enable pytorch profiler",
    )

    parser.add_argument(
        "--gpu_cluster",
        required=False,
        type=str,
        default="gpu-cluster",
        help="name of GPU cluster",
    )
    parser.add_argument(
        "--instance_count",
        required=False,
        type=int,
        default=2,
        help="to increase the number of nodes (distributed training)",
    )
    parser.add_argument(
        "--process_count_per_instance",
        required=False,
        type=int,
        default=1,
        help="how many processes per node (set to number of gpus on instance)",
    )

    return parser


def build(ml_client: MLClient, config: argparse.Namespace) -> Any:
    """Builds the pipeline.

    Args:
        ml_client (MLClient): client to connect to AzureML workspace
        config (argparse.Namespace): configuration options for building the graph

    Returns:
        pipeline_instance (Any): the pipeline instance built, ready to submit
    """
    # load the components from their yaml specifications
    training_func = dsl.load_component(
        yaml_file=os.path.join(
            VISION_COMPONENTS_FOLDER, "torchvision_finetune", "spec.yaml"
        )
    )

    # the dsl decorator tells the sdk that we are defining an Azure ML pipeline
    @dsl.pipeline(
        description="torchvision image classification",  # TODO: document
    )
    def e2e_pipeline(
        train_images,
        valid_images,
    ):
        # the training step is calling our training component with the right arguments
        training_step = training_func(
            # inputs
            train_images=train_images,
            valid_images=valid_images,
            # params (some hardcoded, some given by pipeline parameters)
            num_epochs=config.epochs,
            prefetch_factor=config.prefetch_factor,
            num_workers=config.num_workers,
            enable_profiling=config.enable_profiling,
        )
        # we set the name of the compute target for this training job
        training_step.compute = config.gpu_cluster

        # use process_count_per_instance to parallelize on multiple gpus
        training_step.distribution.process_count_per_instance = (
            config.process_count_per_instance  # set to number of gpus on instance
        )

        # use instance_count to increase the number of nodes (distributed training)
        training_step.resources.instance_count = config.instance_count

        # this pipeline will not return anything
        return {}

    # unpack arguments NAME:VERSION from CLI
    train_images = config.training_dataset.split(":")[0]
    train_images_version = int(config.training_dataset.split(":")[1])
    valid_images = config.validation_dataset.split(":")[0]
    valid_images_version = int(config.validation_dataset.split(":")[1])

    # call the pipeline function to create instance
    pipeline_instance = e2e_pipeline(
        train_images=ml_client.datasets.get(train_images, version=train_images_version),
        valid_images=ml_client.datasets.get(valid_images, version=valid_images_version)
    )

    # return the instance
    return pipeline_instance


if __name__ == "__main__":
    # NOTE: the main() function is imported from pipelines.common.main
    # it will:
    # 1. calls build_arguments() to create an argument parser
    # 2. parse the arguments from the command line
    # 3. connect to Azure ML using MLClient
    # 4. call build() with the right client and config (args)
    # 5. submit the pipeline
    main(build, build_arguments)
