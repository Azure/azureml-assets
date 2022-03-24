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


# resolving a couple useful paths
ASSETS_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
VISION_AREA_ROOT = os.path.abspath(os.path.join(ASSETS_REPO_ROOT, "vision"))
VISION_COMPONENTS_FOLDER = os.path.abspath(
    os.path.join(VISION_AREA_ROOT, "src", "components")
)


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

        # outputs of this pipeline are coded as a dictionary
        # keys can be used to assemble and link this pipeline with other pipelines
        return {}

    # return the instance
    return canary_pipeline()


def main(cli_args: List[str] = None) -> None:
    """Main function (builds and submit pipeline)"""
    parser = argparse.ArgumentParser(__doc__)

    group_aml = parser.add_argument_group(f"Azure ML connection setup")
    group_aml.add_argument(
        "--aml_subscription_id",
        required=False,
        type=str,
        help="(using $AML_SUBSCRIPTION_ID if not provided)",
    )
    group_aml.add_argument(
        "--aml_resource_group_name",
        required=False,
        type=str,
        help="(using $AML_RESOURCE_GROUP_NAME if not provided)",
    )
    group_aml.add_argument(
        "--aml_workspace_name",
        required=False,
        type=str,
        help="(using $AML_WORKSPACE_NAME if not provided)",
    )

    group_aml = parser.add_argument_group(f"Experiment config")
    group_aml.add_argument(
        "--experiment_name", required=False, type=str, default="canary"
    )
    group_aml.add_argument(
        "--validate_only", required=False, action="store_true", default=False
    )
    group_aml.add_argument(
        "--wait_for_completion", required=False, action="store_true", default=False
    )

    group_pipeline = parser.add_argument_group(f"Pipeline config")
    build_arguments(group_pipeline)

    # actually parses arguments now
    args = parser.parse_args(cli_args)  # if cli_args, uses sys.argv

    # get a handle to the workspace
    ml_client = MLClient(
        AzureCliCredential(),
        subscription_id=args.aml_subscription_id
        or os.environ.get("AML_SUBSCRIPTION_ID", None),
        resource_group_name=args.aml_resource_group_name
        or os.environ.get("AML_RESOURCE_GROUP_NAME", None),
        workspace_name=args.aml_workspace_name
        or os.environ.get("AML_WORKSPACE_NAME", None),
    )

    # call build() to get pipeline instance
    pipeline_instance = build(ml_client, args)

    # stop here to just validate (not submit)
    if args.validate_only:
        print("Pipeline built successfully (validation not implemented yet)")
        return

    # submit the pipeline job
    returned_job = ml_client.jobs.create_or_update(
        pipeline_instance,
        # Project's name
        experiment_name=args.experiment_name,
        # If there is no dependency, pipeline run will continue even after the failure of one component
        continue_run_on_step_failure=True,
    )

    # get a URL for the status of the job
    print("********************")
    print("********************")
    print("The url to your job:")
    print("********************")
    print(returned_job.services["Studio"].endpoint)

    # print the pipeline run id
    print("********************")
    print(
        f"The pipeline details can be access programmatically using identifier: {returned_job.name}"
    )
    print("********************")
    print("********************")

    if args.wait_for_completion:
        print("Waiting for job to finish...")
        ml_client.jobs.stream(returned_job.name)


if __name__ == "__main__":
    main()
