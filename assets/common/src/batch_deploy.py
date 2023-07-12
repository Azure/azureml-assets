# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import argparse
import json
import re
import time

from azure.ai.ml import Input
from azure.ai.ml.entities import (
    AmlCompute,
    BatchEndpoint,
    BatchDeployment,
    BatchRetrySettings,
)
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from pathlib import Path

from utils.config import AppName
from utils.common_utils import get_mlclient
from utils.logging_utils import custom_dimensions, get_logger
from utils.exceptions import (
    swallow_all_exceptions,
    BatchEndpointInvocationError,
    EndpointCreationError,
    DeploymentCreationError,
    ComputeCreationError,
)


MAX_INSTANCE_COUNT = 20

logger = get_logger(__name__)
custom_dimensions.app_name = AppName.DEPLOY_MODEL


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # Defaults for batch endpoint has been picked mostly from:
    # https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-batch
    # Some of the defaults have been tweaked to cater to large models.

    # add arguments
    parser.add_argument(
        "--registration_details",
        type=Path,
        help="Json file that contains the ID of registered model to be deployed",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Asset ID of the model registered in workspace/registry.",
    )
    parser.add_argument(
        "--inference_payload",
        type=Path,
        help="Folder containing files used to validate deployment",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the endpoint",
    )
    parser.add_argument("--deployment_name", type=str, help="Name of the the deployment")
    parser.add_argument(
        "--compute_name",
        type=str,
        help="Name of the compute target to execute the batch scoring jobs on",
    )
    parser.add_argument(
        "--size",
        type=str,
        help="Compute instance size to deploy model",
        default="Standard_NC24s_v3",
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=0,
        help="Minimum number of instances of the compute cluster",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=1,
        help="Maximum number of instances of the compute cluster",
    )
    parser.add_argument(
        "--idle_time_before_scale_down",
        type=int,
        default=120,  # 2min
        help="Node Idle Time before scaling down amlCompute",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default="predictions.csv",
        help="Name of the batch scoring output file",
    )
    parser.add_argument(
        "--max_concurrency_per_instance",
        type=int,
        default=1,
        help="The maximum number of parallel scoring_script runs per instance",
    )
    parser.add_argument(
        "--error_threshold",
        type=int,
        default=-1,
        help="The number of file failures that should be ignored",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="The maximum number of retries for a failed or timed-out mini batch",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=500,  # 500sec
        help="The timeout in seconds for scoring a single mini batch.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default="info",
        help="The log verbosity level",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=10,
        help="The number of files the code_configuration.scoring_script can process in one run() call",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        help="The number of nodes to use for each batch scoring job",
        default=1,
        choices=range(1, MAX_INSTANCE_COUNT),
    )
    parser.add_argument(
        "--model_deployment_details",
        type=str,
        help="Json file to which deployment details will be written",
    )
    # parse args
    args = parser.parse_args()
    logger.info(f"Args received {args}")
    print("args received ", args)

    return args


def create_endpoint_and_deployment(ml_client, model_id, endpoint_name, deployment_name, args):
    """Create endpoint and deployment and return details."""
    if(args.compute_name):
        compute_name = args.compute_name
        try:
            compute_cluster = ml_client.compute.get(compute_name)
        except Exception as e:
            compute_cluster = AmlCompute(
                name=compute_name,
                size=args.size,
                min_instances=args.min_instances,
                max_instances=args.max_instances,
                idle_time_before_scale_down=args.idle_time_before_scale_down,
            )
            try:
                print(f"Creating compute cluster {compute_name}")
                ml_client.begin_create_or_update(compute_cluster).wait()
            except Exception as e:
                raise AzureMLException._with_error(
                    AzureMLError.create(ComputeCreationError, exception=e)
                )
    else:
        compute_name = "cpu-cluster"
        compute_cluster = AmlCompute(
            name=compute_name,
            size=args.size,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
            idle_time_before_scale_down=args.idle_time_before_scale_down,
        )
        try:
            print(f"Creating compute cluster {compute_name}")
            ml_client.begin_create_or_update(compute_cluster).wait()
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(ComputeCreationError, exception=e)
            )

    endpoint = BatchEndpoint(name=endpoint_name)

    # deployment
    deployment = BatchDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model_id,
        compute=compute_name,
        output_file_name=args.output_file_name,
        max_concurrency_per_instance=args.max_concurrency_per_instance,
        error_threshold=args.error_threshold,
        retry_settings=BatchRetrySettings(
            max_retries=args.max_retries,
            timeout=args.timeout,
        ),
        logging_level=args.logging_level,
        mini_batch_size=args.mini_batch_size,
        instance_count=args.instance_count,
    )

    try:
        logger.info(f"Creating endpoint {endpoint_name}")
        ml_client.begin_create_or_update(endpoint).wait()
    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(EndpointCreationError, exception=e)
        )

    try:
        logger.info(f"Creating deployment {deployment}")
        ml_client.batch_deployments.begin_create_or_update(deployment).wait()
    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(DeploymentCreationError, exception=e)
        )

    logger.info("Deployment successful.")

    # set the deployment as default
    try:
        logger.info(f"Updating endpoint to make {deployment_name} as default deployment")
        endpoint = ml_client.batch_endpoints.get(endpoint_name)
        endpoint.defaults.deployment_name = deployment_name
        ml_client.begin_create_or_update(endpoint).wait()

        endpoint = ml_client.batch_endpoints.get(endpoint_name)
    except Exception as e:
        error_msg = f"Error occured while updating deployment - {e}"
        raise Exception(error_msg)

    logger.info(f"The default deployment is {endpoint.defaults.deployment_name}")
    return endpoint, deployment


@swallow_all_exceptions(logger)
def main():
    """Run main function."""
    args = parse_args()
    ml_client = get_mlclient()
    # get registered model id
    if args.registration_details:
        model_info = {}
        with open(args.registration_details) as f:
            model_info = json.load(f)
        model_id = model_info["id"]
        model_name = model_info["name"]
    elif args.model_id:
        model_id = str(args.model_id)
        model_name = model_id.split("/")[-3]
    else:
        raise Exception("Arguments model_id and registration_details both are missing.")

    # Endpoint has following restrictions:
    # 1. Name must begin with lowercase letter
    # 2. Followed by lowercase letters, hyphen or numbers
    # 3. End with a lowercase letter or number

    # 1. Replace underscores and slashes by hyphens and convert them to lower case.
    # 2. Take 21 chars from model name and append '-' & timstamp(10chars) to it

    endpoint_name = re.sub("[^A-Za-z0-9]", "-", model_name).lower()[:21]
    endpoint_name = f"{endpoint_name}-{int(time.time())}"
    endpoint_name = endpoint_name

    endpoint_name = args.endpoint_name if args.endpoint_name else endpoint_name
    deployment_name = args.deployment_name if args.deployment_name else "default"

    endpoint, deployment = create_endpoint_and_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_id=model_id,
        args=args
    )

    if args.inference_payload:
        print("Invoking inference with test payload ...")
        try:
            input = Input(type="uri_folder", path=rf"{args.inference_payload}")

            job = ml_client.batch_endpoints.invoke(
                endpoint_name=endpoint.name, input=input
            )

            ml_client.jobs.stream(job.name)

            scoring_job = list(ml_client.jobs.list(parent_job_name=job.name))[0]

            ml_client.jobs.download(
                name=scoring_job.name, download_path=".", output_name="score"
            )

            logger.info("Endpoint invoked successfully.")
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(BatchEndpointInvocationError, exception=e)
            )

    print("Saving deployment details ...")

    # write deployment details to file
    endpoint_type = "aml_online_inference"
    deployment_details = {
        "endpoint_name": endpoint.name,
        "deployment_name": deployment.name,
        "endpoint_uri": endpoint.__dict__["_scoring_uri"],
        "endpoint_type": endpoint_type,
        "size": args.size,
        "instance_count": args.instance_count,
        "max_concurrency_per_instance": args.max_concurrency_per_instance,
    }
    json_object = json.dumps(deployment_details, indent=4)
    with open(args.model_deployment_details, "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved deployment details in output json file.")


# run script
if __name__ == "__main__":
    # run main function
    main()
