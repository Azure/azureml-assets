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


logger = get_logger(__name__)
custom_dimensions.app_name = AppName.TGI_DEPLOY_MODEL

CODE = "tgi_score"
SCORE = "score.py"
DEPLOYMENT_ENV_ASSET_URI = "azureml://registries/azureml/environments/foundation-model-inference/versions/12"

AZUREML_LOG_LEVEL = "AZUREML_LOG_LEVEL"
LOG_IO = "LOG_IO"
WORKER_TIMEOUT = "WORKER_TIMEOUT"

# model init
MODEL_INIT_ARGS = "model_init_args"
MLMODEL_PATH = "MLMODEL_PATH"
MODEL_ID = "MODEL_ID"
SHARDED = "SHARDED"
NUM_SHARD = "NUM_SHARD"
QUANTIZE = "QUANTIZE"
DTYPE = "DTYPE"
TRUST_REMOTE_CODE = "TRUST_REMOTE_CODE"
MAX_CONCURRENT_REQUESTS = "MAX_CONCURRENT_REQUESTS"
MAX_BEST_OF = "MAX_BEST_OF"
MAX_STOP_SEQUENCES = "MAX_STOP_SEQUENCES"
MAX_INPUT_LENGTH = "MAX_INPUT_LENGTH"
MAX_TOTAL_TOKENS = "MAX_TOTAL_TOKENS"
# client init
CLIENT_INIT_ARGS = "client_init_args"
CLIENT_TIMEOUT = "TIMEOUT"


def create_endpoint_and_deployment(ml_client, model_id, endpoint_name, deployment_name, env_vars, args):
    """Create endpoint and deployment and return details."""
    logger.info(f"Creating endpoint with endpoint name: {endpoint_name} deployment_name {deployment_name}")
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

    logger.info(f"Deployment successful. Updating endpoint to take 100% traffic for deployment {deployment_name}")

    # deployment to take 100% traffic
    endpoint.traffic = {deployment.name: 100}
    try:
        ml_client.begin_create_or_update(endpoint).wait()
        endpoint = ml_client.online_endpoints.get(endpoint.name)
        logger.info("Done \u2713")
    except Exception as e:
        error_msg = f"Error occured while updating endpoint traffic - {e}"
        logging.error(error_msg)
        raise Exception(error_msg)

    logger.info(f"Endpoint updated to take 100% traffic for deployment {deployment_name}")
    return endpoint, deployment


def parse_args():
    """Return arguments."""
    parser = argparse.ArgumentParser()
    # model details params
    parser.add_argument("--model_id", type=str, required=False, help="registered model asset id")
    parser.add_argument(
        "--registration_details_folder",
        type=Path,
        help="Folder containing model registration details in a JSON file named model_registration_details.json",
    )

    # tgl params
    parser.add_argument("--tgl_num_shard", type=int, required=False)
    parser.add_argument("--tgl_sharded", type=bool, required=False)
    parser.add_argument("--tgl_quantize", type=str, required=False)
    parser.add_argument("--tgl_dtype", type=str, required=False)
    parser.add_argument("--tgl_trust_remote_code", type=bool, required=False)
    parser.add_argument("--tgl_max_concurrent_requests", type=int, required=False)
    parser.add_argument("--tgl_max_best_of", type=int, required=False)
    parser.add_argument("--tgl_max_stop_sequences", type=int, required=False)
    parser.add_argument("--tgl_max_input_length", type=int, required=False)
    parser.add_argument("--tgl_max_total_tokens", type=int, required=False)

    # inference params
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

    # tgl params
    tgl_num_shard = args.tgl_num_shard
    tgl_sharded = args.tgl_sharded
    tgl_quantize = args.tgl_quantize
    tgl_dtype = args.tgl_dtype
    tgl_trust_remote_code = args.tgl_trust_remote_code
    tgl_max_concurrent_requests = args.tgl_max_concurrent_requests
    tgl_max_best_of = args.tgl_max_best_of
    tgl_max_stop_sequences = args.tgl_max_stop_sequences
    tgl_max_input_length = args.tgl_max_input_length
    tgl_max_total_tokens = args.tgl_max_total_tokens

    if tgl_sharded and tgl_num_shard and tgl_num_shard <= 1:
        raise Exception("Num shard should be greater than 1 if sharded is set")

    if not tgl_sharded and not tgl_num_shard:
        logger.info("Setting NUM_SHARD to 1 for SHARDED and NUM_SHARD not provided.")

    if tgl_quantize and tgl_dtype:
        raise Exception("dtype and quantize does not work together.")

    # model params
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
    endpoint_name = f"tgi-{endpoint_name}-{int(time.time())}"
    endpoint_name = endpoint_name

    endpoint_name = args.endpoint_name if args.endpoint_name else endpoint_name
    deployment_name = args.deployment_name if args.deployment_name else "default"

    huggingface_model_path = "mlflow_model_folder/data/model"
    mlmodel_path = "mlflow_model_folder/MLmodel"

    client_init_args = {CLIENT_TIMEOUT: args.request_timeout_ms / 1000}
    model_init_args = {MODEL_ID: huggingface_model_path, MLMODEL_PATH: mlmodel_path}

    if tgl_num_shard:
        model_init_args[NUM_SHARD] = str(tgl_num_shard)
    if tgl_sharded:
        model_init_args[SHARDED] = str(tgl_sharded).lower()
    if tgl_quantize:
        model_init_args[QUANTIZE] = str(tgl_quantize)
    if tgl_dtype:
        model_init_args[DTYPE] = str(tgl_dtype)
    if tgl_trust_remote_code:
        model_init_args[TRUST_REMOTE_CODE] = str(tgl_trust_remote_code).lower()
    if tgl_max_concurrent_requests:
        model_init_args[MAX_CONCURRENT_REQUESTS] = str(tgl_max_concurrent_requests)
    if tgl_max_best_of:
        model_init_args[MAX_BEST_OF] = str(tgl_max_best_of)
    if tgl_max_stop_sequences:
        model_init_args[MAX_STOP_SEQUENCES] = str(tgl_max_stop_sequences)
    if tgl_max_input_length:
        model_init_args[MAX_INPUT_LENGTH] = str(tgl_max_input_length)
    if tgl_max_total_tokens:
        model_init_args[MAX_TOTAL_TOKENS] = str(tgl_max_total_tokens)

    env_vars = {
        AZUREML_LOG_LEVEL: DeployConstants.LOG_LEVEL,
        LOG_IO: DeployConstants.LOG_IO,
        WORKER_TIMEOUT: DeployConstants.WORKER_TIMEOUT,
        **model_init_args,
        **client_init_args,
    }

    mlclient = get_mlclient()
    endpoint, deployment = create_endpoint_and_deployment(
        mlclient=mlclient,
        endpoint_name=endpoint_name,
        deployment_name=deployment_name,
        model_id=model_id,
        env_vars=env_vars,
        args=args
    )

    if args.inference_payload:
        logger.info("Invoking inference with test payload ...")
        try:
            response = mlclient.online_endpoints.invoke(
                endpoint_name=endpoint_name,
                deployment_name=deployment_name,
                request_file=args.inference_payload,
            )
            logger.info(f"Response:\n{response}")
        except Exception as e:
            raise AzureMLException._with_error(
                AzureMLError.create(OnlineEndpointInvocationError, exception=e)
            )

    logger.info("Saving deployment details")
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
    main()
