import os
import sys
import argparse

# handle to the workspace
from azure.ml import MLClient

# authentication package
from azure.identity import AzureCliCredential

# python sdk
from azure.ml import dsl

# get a handle to the workspace
ml_client = MLClient(
    AzureCliCredential(),
    # subscription_id="<SUBSCRIPTION_ID>",
    # resource_group_name="<RESOURCE_GROUP>",
    # workspace_name="<AML_WORKSPACE_NAME>",
)

# resolving a couple useful paths
ASSETS_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
VISION_AREA_ROOT = os.path.abspath(os.path.join(ASSETS_REPO_ROOT, "vision"))
VISION_COMPONENTS_FOLDER = os.path.abspath(
    os.path.join(VISION_AREA_ROOT, "src", "components")
)

##################################################################
### PIPELINE DEFINITION
##################################################################

# load the component from its yaml specifications
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
    generate_random_step.compute = "cpu-cluster"

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
    training_step.compute = "gpu-cluster"

    # use process_count_per_instance to parallelize on multiple gpus
    training_step.distribution.process_count_per_instance = (
        1  # set to number of gpus on instance
    )

    # use instance_count to increase the number of nodes (machines)
    training_step.resources.instance_count = 1

    # outputs of this pipeline are coded as a dictionary
    # keys can be used to assemble and link this pipeline with other pipelines
    return {}


##################################################################
### RUN / MAIN
##################################################################

def main(cli_args: list = None) -> None:
    """Running the pipeline"""
    # create instance
    pipeline_instance = canary_pipeline()

    # submit the pipeline job
    returned_job = ml_client.jobs.create_or_update(
        pipeline_instance,
        # Project's name
        experiment_name="canary",
        # If there is no dependency, pipeline run will continue even after the failure of one component
        continue_run_on_step_failure=True,
    )

    # get a URL for the status of the job
    print("The url to see your live job running is returned by the sdk:")
    print(returned_job.services["Studio"].endpoint)

    # print the pipeline run id
    print(
        f"The pipeline details can be access programmatically using identifier: {returned_job.name}"
    )

if __name__ == "__main__":
    main()
