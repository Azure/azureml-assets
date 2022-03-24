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
    model_name,  # a name to register the model after training
    epochs,  # the number of epochs
    enable_profiling,  # bonus: we've implemented pytorch profiling in our script
):
    # the training step is calling our training component with the right arguments
    training_step = training_func(
        # inputs
        train_images=train_images,
        valid_images=valid_images,
        # params (some hardcoded, some given by pipeline parameters)
        num_epochs=epochs,
        register_model_as=model_name,
        num_workers=-1,  # use all cpus (see train.py)
        enable_profiling=enable_profiling,  # turns on profiler (see train.py)
    )
    # we set the name of the compute target for this training job
    training_step.compute = "gpu-cluster"

    # use process_count_per_instance to parallelize on multiple gpus
    training_step.distribution.process_count_per_instance = (
        1  # set to number of gpus on instance
    )

    # use instance_count to increase the number of nodes (machines)
    training_step.resources.instance_count = 8

    # outputs of this pipeline are coded as a dictionary
    # keys can be used to assemble and link this pipeline with other pipelines
    return {"model": training_step.outputs.trained_model}


##################################################################
### RUN / MAIN
##################################################################


def run(args: argparse.Namespace) -> None:
    """Running the pipeline"""
    # create instance
    pipeline_instance = e2e_pipeline(
        # inputs: using validation set for training makes model invalid
        train_images=ml_client.datasets.get("places2_train", version=1),
        valid_images=ml_client.datasets.get("places2_valid", version=1),
        model_name="places",
        epochs=10,
        enable_profiling=False,
    )

    # submit the pipeline job
    returned_job = ml_client.jobs.create_or_update(
        pipeline_instance,
        # Project's name
        experiment_name="e2e_image_classification_sample",
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


def main(cli_args: list = None) -> None:
    """Main function"""
    run(None)


if __name__ == "__main__":
    main()
