# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run MII Model deployment module."""

import argparse
import json
import logging
import re
import time
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
    CodeConfiguration,
    ProbeSettings,
)
from azureml._common._error_definition import AzureMLError
from azureml._common.exceptions import AzureMLException
from pathlib import Path
from utils.common_utils import get_mlclient, get_model_name
from utils.config import AppName, ComponentVariables, DeployConstants
from utils.logging_utils import custom_dimensions, get_logger
from utils.exceptions import (
    swallow_all_exceptions,
    OnlineEndpointInvocationError,
    EndpointCreationError,
    DeploymentCreationError,
)


CODE = "dsmii_score"
SCORE = "score.py"
DEPLOYMENT_ENV_ASSET_URI = "azureml://registries/azureml/environments/foundation-model-inference/versions/10"

AZUREML_LOG_LEVEL = "AZUREML_LOG_LEVEL"
LOG_IO = "LOG_IO"
WORKER_COUNT = "WORKER_COUNT"
WORKER_TIMEOUT = "WORKER_TIMEOUT"
MAX_TOTAL_TOKENS = "MAX_TOTAL_TOKENS"
TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"

logger = get_logger(__name__)
custom_dimensions.app_name = AppName.MII_DEPLOY_MODEL


def create_endpoint_and_deployment(ml_client, model_id, endpoint_name, deployment_name, env_vars, args):
    """Create endpoint and deployment and return details."""
    print(f"Creating endpoint with endpoint name: {endpoint_name} deployment_name {deployment_name}")
    endpoint = ManagedOnlineEndpoint(name=endpoint_name, auth_mode="key")

    code_configuration = CodeConfiguration(code=CODE, scoring_script=SCORE)
    request_settings = OnlineRequestSettings(
        max_concurrent_requests_per_instance=args.max_concurrent_requests_per_instance,
        request_timeout_ms=args.request_timeout_ms,
        max_queue_wait_ms=args.max_queue_wait_ms,
    )

    liveness_probe_settings = ProbeSettings(
        failure_threshold=args.failure_threshold_liveness_probe,
        timeout=args.timeout_liveness_probe,
        period=args.period_liveness_probe,
        initial_delay=args.initial_delay_liveness_probe,
    )

    readiness_probe_settings = ProbeSettings(
        failure_threshold=args.failure_threshold_readiness_probe,
        success_threshold=args.success_threshold_readiness_probe,
        timeout=args.timeout_readiness_probe,
        period=args.period_readiness_probe,
        initial_delay=args.initial_delay_readiness_probe,
    )

    # deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        code_configuration=code_configuration,
        endpoint_name=endpoint_name,
        environment=DEPLOYMENT_ENV_ASSET_URI,
        environment_variables=env_vars,
        model=model_id,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        request_settings=request_settings,
        liveness_probe=liveness_probe_settings,
        readiness_probe=readiness_probe_settings,
        egress_public_network_access=args.egress_public_network_access,
    )

    try:
        logger.info(f"Creating endpoint {endpoint_name}")
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
        logger.info(f"Endpoint created {endpoint.id}")
    except Exception as e:
        raise AzureMLException._with_error(
            AzureMLError.create(EndpointCreationError, exception=e)
        )

    try:
        logger.info(f"Creating deployment {deployment}")
        ml_client.online_deployments.begin_create_or_update(deployment).wait()
    except Exception as e:
        try:
            logger.error("Deployment failed. Printing deployment logs")
            logs = ml_client.online_deployments.get_logs(
                name=deployment_name,
                endpoint_name=endpoint_name,
                lines=DeployConstants.MAX_DEPLOYMENT_LOG_TAIL_LINES
            )
            logger.error(logs)
        except Exception as ex:
            logger.error(f"Error in fetching deployment logs: {ex}")

        raise AzureMLException._with_error(
            AzureMLError.create(DeploymentCreationError, exception=e)
        )

    print(f"Deployment successful. Updating endpoint to take 100% traffic for deployment {deployment_name}")

    # deployment to take 100% traffic
    endpoint.traffic = {deployment.name: 100}
    try:
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
    except Exception as e:
        error_msg = f"Error occured while updating endpoint traffic. Deployment should be usable. Exception - {e}"
        raise Exception(error_msg)

    logger.info(f"Endpoint updated to take 100% traffic for deployment {deployment_name}")
    return endpoint, deployment


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mii_max_new_tokens", type=str, required=False, default=4096, help="MAX TOKENS to set during model init")
    parser.add_argument("--mii_replica_num", type=str, required=False, default=1, help="Replica num")
    parser.add_argument("--mii_trust_remote_code", type=bool, required=False, default=True, help="Trust remote code")

    parser.add_argument("--model_id", type=str, required=False, help="registered model asset id")
    parser.add_argument(
        "--registration_details_folder",
        type=Path,
        help="Folder containing model registration details in a JSON file named model_registration_details.json",
    )

    parser.add_argument(
        "--inference_payload",
        type=Path,
        help="Json file with inference endpoint payload.",
    )
    parser.add_argument(
        "--endpoint_name",
        type=str,
        help="Name of the endpoint",
    )
    parser.add_argument("--deployment_name", type=str, help="Name of the the deployment")
    parser.add_argument(
        "--instance_type",
        type=str,
        help="Compute instance type to deploy model",
        default="Standard_NC24s_v3",
    )
    parser.add_argument(
        "--instance_count",
        type=int,
        help="Number of compute instances to deploy model",
        default=1,
        choices=range(1, DeployConstants.MAX_INSTANCE_COUNT),
    )
    parser.add_argument(
        "--max_concurrent_requests_per_instance",
        type=int,
        default=1,
        help="Maximum concurrent requests to be handled per instance",
    )
    parser.add_argument(
        "--request_timeout_ms",
        type=int,
        default=DeployConstants.MAX_REQUEST_TIMEOUT,
        help="Request timeout in ms.",
    )
    parser.add_argument(
        "--max_queue_wait_ms",
        type=int,
        default=DeployConstants.DEFAULT_REQUEST_QUEUE_WAIT,
        help="Maximum queue wait time of a request in ms",
    )

    parser.add_argument(
        "--failure_threshold_liveness_probe",
        type=int,
        default=DeployConstants.PROBE_FAILURE_THRESHOLD,
        help="No of times system will try after failing the liveness probe",
    )
    parser.add_argument(
        "--timeout_liveness_probe",
        type=int,
        default=DeployConstants.PROBE_TIMEOUT,
        help="The number of seconds after which the liveness probe times out",
    )
    parser.add_argument(
        "--period_liveness_probe",
        type=int,
        default=DeployConstants.PROBE_PERIOD,
        help="How often (in seconds) to perform the liveness probe",
    )
    parser.add_argument(
        "--initial_delay_liveness_probe",
        type=int,
        default=DeployConstants.PROBE_INITIAL_DELAY,
        help="The number of seconds after the container has started before the liveness probe is initiated",
    )

    parser.add_argument(
        "--failure_threshold_readiness_probe",
        type=int,
        default=DeployConstants.PROBE_FAILURE_THRESHOLD,
        help="No of times system will try after failing the readiness probe",
    )
    parser.add_argument(
        "--success_threshold_readiness_probe",
        type=int,
        default=DeployConstants.PROBE_SUCCESS_THRESHOLD,
        help="The minimum consecutive successes for the readiness probe to be considered successful, after fail",
    )
    parser.add_argument(
        "--timeout_readiness_probe",
        type=int,
        default=DeployConstants.PROBE_TIMEOUT,
        help="The number of seconds after which the readiness probe times out",
    )
    parser.add_argument(
        "--period_readiness_probe",
        type=int,
        default=DeployConstants.PROBE_PERIOD,
        help="How often (in seconds) to perform the readiness probe",
    )
    parser.add_argument(
        "--initial_delay_readiness_probe",
        type=int,
        default=DeployConstants.PROBE_INITIAL_DELAY,
        help="The number of seconds after the container has started before the readiness probe is initiated",
    )

    parser.add_argument(
        "--egress_public_network_access",
        type=str,
        default="enabled",
        help="Secures the deployment by restricting interaction between deployment and Azure resources used by it",
    )
    parser.add_argument(
        "--model_deployment_details",
        type=str,
        help="Json file to which deployment details will be written",
    )

    args = parser.parse_args()
    logging.info(f"Args received {args}")

    # Validating passed input values
    if args.max_concurrent_requests_per_instance < 1:
        raise Exception("Arg max_concurrent_requests_per_instance cannot be less than 1")
    if args.request_timeout_ms < 1 or args.request_timeout_ms > DeployConstants.MAX_REQUEST_TIMEOUT:
        raise Exception(f"Arg request_timeout_ms should lie between 1 and {DeployConstants.MAX_REQUEST_TIMEOUT}")
    if args.max_queue_wait_ms < 1 or args.max_queue_wait_ms > DeployConstants.MAX_REQUEST_TIMEOUT:
        raise Exception(f"Arg max_queue_wait_ms should lie between 1 and {DeployConstants.MAX_REQUEST_TIMEOUT}")

    return args


@swallow_all_exceptions(logger)
def main():
    args = parse_args()

    mii_max_new_tokens = args.mii_max_new_tokens
    mii_replica_num = args.mii_replica_num
    mii_trust_remote_code = args.mii_trust_remote_code

    model_id = args.model_id
    model_deployment_details = args.model_deployment_details

    if args.model_id:
        model_id = str(args.model_id)
    elif args.registration_details_folder:
        registration_details_file = args.registration_details_folder / ComponentVariables.REGISTRATION_DETAILS_JSON_FILE
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
    endpoint_name = re.sub('[^A-Za-z0-9]', '-', model_name).lower()[:17]
    endpoint_name = f"mii-{endpoint_name}-{int(time.time())}"
    endpoint_name = endpoint_name

    endpoint_name = args.endpoint_name if args.endpoint_name else endpoint_name
    deployment_name = args.deployment_name if args.deployment_name else "default"

    env_vars = {
        WORKER_COUNT: mii_replica_num,
        MAX_TOTAL_TOKENS: mii_max_new_tokens,
        TRUST_REMOTE_CODE: mii_trust_remote_code,
        AZUREML_LOG_LEVEL: DeployConstants.LOG_LEVEL,
        LOG_IO: DeployConstants.LOG_IO,
        WORKER_TIMEOUT: DeployConstants.WORKER_TIMEOUT,
    }

    mlclient = get_mlclient()
    endpoint, deployment = create_endpoint_and_deployment(
        mlclient=mlclient,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_id=model_id,
        env_vars=env_vars,
        args=args,
    )

    if args.inference_payload:
        print("Invoking inference with test payload ...")
        try:
            response = mlclient.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=args.inference_payload,
            )
            print(f"Response:\n{response}")
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(OnlineEndpointInvocationError, exception=e)
            )

    print("Saving deployment details")
    # write deployment details to file
    endpoint_type = "aml_online_inference"
    deployment_details = {
        "endpoint_name": endpoint.name,
        "deployment_name": deployment.name,
        "endpoint_uri": endpoint.__dict__["_scoring_uri"],
        "endpoint_type": endpoint_type,
        "instance_type": args.instance_type,
        "instance_count": args.instance_count,
        "max_concurrent_requests_per_instance": args.max_concurrent_requests_per_instance,
    }

    json_object = json.dumps(deployment_details, indent=4)
    with open(model_deployment_details, "w") as outfile:
        outfile.write(json_object)
    logger.info("Saved deployment details in output json file.")


if __name__ == "__main__":
    # run main function
    main()
