# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Model deployment module."""
import argparse
import json
import re
import time
import shutil

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

from utils.config import AppName, ComponentVariables
from utils.common_utils import get_mlclient, get_model_name
from utils.logging_utils import custom_dimensions, get_logger
from utils.exceptions import (
    swallow_all_exceptions,
    BatchEndpointInvocationError,
    EndpointCreationError,
    DeploymentCreationError,
    ComputeCreationError,
)


MAX_INSTANCE_COUNT = 20
DEFAULT_COMPUTE_SIZE = "Standard_NC24s_v3"
DEFAULT_MIN_INSTANCES = 0
DEFAULT_MAX_INSTANCES = 1
DEFAULT_IDLE_TIME_BEFORE_SCALE_DOWN = 120  # 2min
DEFAULT_OUTPUT_FILE_NAME = "predictions.csv"
DEFAULT_MAX_CONCURRENCY_PER_INSTANCE = 1
DEFAULT_ERROR_THRESHOLD = -1
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 500  # 500sec
DEFAULT_LOGGING_LEVEL = "info"
DEFAULT_MINI_BATCH_SIZE = 10
DEFAULT_INSTANCE_COUNT = 1
DEPLOYMENT_DETAILS_JSON_FILE_NAME = "model_deployment_details.json"
NAMED_OUTPUTS_FOLDER = "named-outputs"
SCORE_FOLDER = "score"


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.BATCH_DEPLOY_MODEL


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()

    # Defaults for batch endpoint has been picked mostly from:
    # https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-deployment-batch
    # Some of the defaults have been tweaked to cater to large models.

    # add arguments
    parser.add_argument(
        "--registration_details_folder",
        type=Path,
        help="Folder containing model registration details in a JSON file named model_registration_details.json",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        help="Asset ID of the model registered in workspace/registry.",
    )
    parser.add_argument(
        "--inference_payload_file",
        type=Path,
        help="File containing data used to validate deployment",
    )
    parser.add_argument(
        "--inference_payload_folder",
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
        default=DEFAULT_COMPUTE_SIZE,
    )
    parser.add_argument(
        "--min_instances",
        type=int,
        default=DEFAULT_MIN_INSTANCES,
        help="Minimum number of instances of the compute cluster",
    )
    parser.add_argument(
        "--max_instances",
        type=int,
        default=DEFAULT_MAX_INSTANCES,
        help="Maximum number of instances of the compute cluster",
    )
    parser.add_argument(
        "--idle_time_before_scale_down",
        type=int,
        default=DEFAULT_IDLE_TIME_BEFORE_SCALE_DOWN,
        help="Node Idle Time before scaling down amlCompute",
    )
    parser.add_argument(
        "--output_file_name",
        type=str,
        default=DEFAULT_OUTPUT_FILE_NAME,
        help="Name of the batch scoring output file",
    )
    parser.add_argument(
        "--max_concurrency_per_instance",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY_PER_INSTANCE,
        help="The maximum number of parallel scoring_script runs per instance",
    )
    parser.add_argument(
        "--error_threshold",
        type=int,
        default=DEFAULT_ERROR_THRESHOLD,
        help="The number of file failures that should be ignored",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="The maximum number of retries for a failed or timed-out mini batch",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="The timeout in seconds for scoring a single mini batch.",
    )
    parser.add_argument(
        "--logging_level",
        type=str,
        default=DEFAULT_LOGGING_LEVEL,
        help="The log verbosity level",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=DEFAULT_MINI_BATCH_SIZE,
        help="The number of files the code_configuration.scoring_script can process in one run() call",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        help="The number of nodes to use for each batch scoring job",
        default=DEFAULT_INSTANCE_COUNT,
        choices=range(1, MAX_INSTANCE_COUNT),
    )
    parser.add_argument(
        "--batch_job_output_folder",
        type=Path,
        help="Folder to which batch job outputs will be saved",
    )
    # parse args
    args = parser.parse_args()
    logger.info(f"Args received {args}")
    print("args received ", args)

    return args


def download_batch_output(ml_client, job, args):
    """Download the output file."""
    scoring_job = list(ml_client.jobs.list(parent_job_name=job.name))[0]

    logger.info(f"Downloading the {args.output_file_name} file.")
    ml_client.jobs.download(
        name=scoring_job.name, download_path=args.batch_job_output_folder, output_name="score"
    )

    source_folder = args.batch_job_output_folder / NAMED_OUTPUTS_FOLDER / SCORE_FOLDER
    destination_folder = args.batch_job_output_folder / SCORE_FOLDER

    shutil.copytree(source_folder, destination_folder)

    shutil.rmtree(args.batch_job_output_folder / NAMED_OUTPUTS_FOLDER)
    logger.info(f"Successfully downloaded the {args.output_file_name} file.")


def invoke_endpoint_job(ml_client, endpoint, type, args):
    """Invoke a job using the endpoint."""
    print(f"Invoking inference with {type} test payload ...")
    try:
        if type == "uri_folder":
            path = args.inference_payload_folder
        else:
            path = args.inference_payload_file
        input = Input(type=type, path=rf"{path}")

        job = ml_client.batch_endpoints.invoke(
            endpoint_name=endpoint.name, input=input
        )

        ml_client.jobs.stream(job.name)
        logger.info(f"Endpoint invoked successfully with {type} test payload.")
        download_batch_output(ml_client, job, args)

    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(BatchEndpointInvocationError, exception=e)
        )


def get_or_create_compute(ml_client, compute_name, args):
    """Get or create the compute cluster and return details."""
    try:
        compute_cluster = ml_client.compute.get(compute_name)
        logger.info(f"Using compute cluster {compute_name}.")
    except Exception:
        compute_cluster = AmlCompute(
            name=compute_name,
            size=args.size,
            min_instances=args.min_instances,
            max_instances=args.max_instances,
            idle_time_before_scale_down=args.idle_time_before_scale_down,
        )
        try:
            logger.info(f"Creating compute cluster {compute_name}")
            ml_client.begin_create_or_update(compute_cluster).wait()
            logger.info("Compute cluster created successfully.")
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(ComputeCreationError, exception=e)
            )
    return compute_cluster


def create_endpoint_and_deployment(ml_client, compute_name, model_id, endpoint_name, deployment_name, args):
    """Create endpoint and deployment and return details."""
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
        logger.info("Endpoint created successfully.")
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

    if args.model_id:
        model_id = str(args.model_id)
    elif args.registration_details_folder:
        registration_details_file = args.registration_details_folder/ComponentVariables.REGISTRATION_DETAILS_JSON_FILE
        if registration_details_file.exists():
            try:
                with open(registration_details_file) as f:
                    model_info = json.load(f)
                model_id = model_info["id"]
            except Exception as e:
                raise Exception(f"model_registration_details json file is missing model information {e}.")
        else:
            raise Exception(f"{ComponentVariables.REGISTRATION_DETAILS_JSON_FILE} is missing inside folder.")
    else:
        raise Exception("Arguments model_id and registration_details both are missing.")

    # Endpoint has following restrictions:
    # 1. Name must begin with lowercase letter
    # 2. Followed by lowercase letters, hyphen or numbers
    # 3. End with a lowercase letter or number

    # 1. Replace underscores and slashes by hyphens and convert them to lower case.
    # 2. Take 21 chars from model name and append '-' & timstamp(10chars) to it
    model_name = get_model_name(model_id)

    endpoint_name = re.sub("[^A-Za-z0-9]", "-", model_name).lower()[:21]
    endpoint_name = f"{endpoint_name}-{int(time.time())}"
    endpoint_name = endpoint_name

    endpoint_name = args.endpoint_name if args.endpoint_name else endpoint_name
    deployment_name = args.deployment_name if args.deployment_name else "default"
    compute_name = args.compute_name if args.compute_name else "cpu-cluster"

    compute_cluster = get_or_create_compute(
        ml_client=ml_client,
        compute_name=compute_name,
        args=args
    )

    endpoint, deployment = create_endpoint_and_deployment(
        ml_client=ml_client,
        compute_name=compute_name,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_id=model_id,
        args=args
    )

    if args.inference_payload_file and args.inference_payload_folder:
        logger.warning("Dump all csv files under uri_folder instead of providing multiple inputs.")

    if args.inference_payload_folder:
        invoke_endpoint_job(
            ml_client=ml_client,
            endpoint=endpoint,
            type="uri_folder",
            args=args,
        )
    elif args.inference_payload_file:
        invoke_endpoint_job(
            ml_client=ml_client,
            endpoint=endpoint,
            type="uri_file",
            args=args,
        )

    print("Saving deployment details ...")

    # write deployment details to file
    endpoint_type = "aml_batch_inference"
    deployment_details = {
        "endpoint_name": endpoint.name,
        "deployment_name": deployment.name,
        "endpoint_uri": endpoint.__dict__["_scoring_uri"],
        "endpoint_type": endpoint_type,
        "compute_size": compute_cluster.size,
        "instance_count": deployment.instance_count,
        "max_concurrency_per_instance": deployment.max_concurrency_per_instance,
    }
    json_object = json.dumps(deployment_details, indent=4)
    file_path = args.batch_job_output_folder / DEPLOYMENT_DETAILS_JSON_FILE_NAME
    with open(file_path, "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved deployment details in output json file.")


# run script
if __name__ == "__main__":
    # run main function
    main()
