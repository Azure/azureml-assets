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
SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if SRC_ROOT not in sys.path:
    print(f"Adding {SRC_ROOT} to path")
    sys.path.append(str(SRC_ROOT))

from pipelines.common.main import main, VISION_COMPONENTS_FOLDER


def build_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Constructs the argument parser"""
    parser.add_argument(
        "--cpu_cluster",
        required=False,
        type=str,
        default="cpu-cluster",
        help="name of CPU cluster",
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
    generate_func = dsl.load_component(
        yaml_file=os.path.join(
            VISION_COMPONENTS_FOLDER, "generate_random_image_classes", "spec.yaml"
        )
    )
    training_func = dsl.load_component(
        yaml_file=os.path.join(
            VISION_COMPONENTS_FOLDER, "torchvision_finetune", "spec.yaml"
        )
    )

    # the dsl decorator tells the sdk that we are defining an Azure ML pipeline
    @dsl.pipeline(
        description="canary pipeline image classification",  # TODO: document
    )
    def canary_pipeline():
        # generate random data
        generate_random_step = generate_func(
            classes=4,
            train_samples=100,
            valid_samples=100,
            width=300,
            height=300,
        )
        # we set the name of the compute target for this training job
        generate_random_step.compute = config.cpu_cluster

        # the training step is calling our training component with the right arguments
        training_step = training_func(
            # inputs
            train_images=generate_random_step.outputs.output_train,
            valid_images=generate_random_step.outputs.output_valid,
            # params (hardcoded for canary)
            num_epochs=1,
            num_workers=0,  # do not use prefetching
            enable_profiling=False,  # turns profiler off
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

    # return the instance
    return canary_pipeline()


if __name__ == "__main__":
    # NOTE: the main() function is imported from pipelines.common.main
    # it will:
    # 1. calls build_arguments() to create an argument parser
    # 2. parse the arguments from the command line
    # 3. connect to Azure ML using MLClient
    # 4. call build() with the right client and config (args)
    # 5. submit the pipeline
    main(build, build_arguments)
