import os
import sys
import argparse
from typing import List, Any

# handle to the workspace
from azure.ml import MLClient

# authentication package
from azure.identity import AzureCliCredential

# resolving a couple useful paths
ASSETS_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
VISION_AREA_ROOT = os.path.abspath(os.path.join(ASSETS_REPO_ROOT, "vision"))
VISION_COMPONENTS_FOLDER = os.path.abspath(
    os.path.join(VISION_AREA_ROOT, "src", "components")
)

def main(build_function, build_arguments_function, cli_args: List[str] = None) -> None:
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
    build_arguments_function(group_pipeline)

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

    # call build_function() to get pipeline instance
    pipeline_instance = build_function(ml_client, config=args)

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
