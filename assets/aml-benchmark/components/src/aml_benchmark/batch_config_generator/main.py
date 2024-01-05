# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch benchmark config generator component."""

import argparse
import json
from enum import Enum
from typing import Dict, Optional

from aml_benchmark.utils.io import save_json_to_file
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.aml_run_utils import str2bool
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError
from aml_benchmark.utils.exceptions import BenchmarkUserException


logger = get_logger(__name__)


class AuthenticationType(Enum):
    """Authentication Type enum for endpoints."""
    AZUREML_WORKSPACE_CONNECTION = "azureml_workspace_connection"
    MANAGED_IDENTITY = "managed_identity"

class ModelType(Enum):
    """Model Type enum."""
    OAI = "oai"
    OSS = "oss"
    VISION_OSS = "vision_oss"


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--scoring_url",
        type=str,
        help="The URL of the endpoint."
    )
    parser.add_argument(
        "--connection_name",
        type=str,
        help="The name of the connection to fetch the API_KEY for the endpoint authentication.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--authentication_type",
        type=str,
        help="Authentication type for endpoint.",
        choices=list(AuthenticationType)
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of model.",
        choices=list(ModelType)
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        help="The deployment name. Only needed for managed OSS deployment.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--debug_mode",
        type=str2bool,
        help="Enable debug mode will print all the debug logs in the score step."
    )
    parser.add_argument(
        "--additional_headers",
        type=str,
        help="A stringified json expressing additional headers to be added to each request.",
        default=None,
        required=False
    )
    parser.add_argument(
        "--ensure_ascii",
        type=str2bool,
        help="If true, the output will have all non-ASCII characters escaped.",
        default=False,
        required=False
    )
    parser.add_argument(
        "--max_retry_time_interval",
        type=int,
        help="The maximum time (in seconds) spent retrying a payload. If unspecified, payloads are retried unlimited times.",
        required=False,
        default=None
    )
    parser.add_argument(
        "--initial_worker_count",
        type=int,
        help="The initial number of workers to use for scoring."
    )
    parser.add_argument(
        "--max_worker_count",
        type=int,
        help="Overrides initial_worker_count if necessary."
    )
    parser.add_argument(
        "--batch_score_config_path",
        type=str,
        help="The config json file for the batch score component."
    )
    parser.add_argument(
        "--response_segment_size",
        type=int,
        help="The maximum number of tokens to generate at a time."
    )
    arguments, _ = parser.parse_known_args()
    logger.info(f"Arguments: {arguments}")
    return arguments


def _get_authentication_config(
    authentication_type: AuthenticationType,
    connection_name: Optional[str]
) -> Dict[str, str]:
    """
    Get authentication config for the provided authentication type and connection name.
    :param authentication_type: Authentication type for endpoint.
    :param connection_name: The name of the connection to fetch the API_KEY for the
        endpoint authentication.
    :return: The dictionary representing the authentication config.
    """
    if authentication_type is AuthenticationType.AZUREML_WORKSPACE_CONNECTION:
        if not connection_name:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details="Connection name should be provided \
                        when authentication_type is set to azureml_workspace_connection."
                )
            )
        return {
            "type": "connection",
            "name": connection_name
        }
    elif authentication_type is AuthenticationType.MANAGED_IDENTITY:
        return {
            "type": "managed_identity",
        }
    raise BenchmarkUserException._with_error(
        AzureMLError.create(
            BenchmarkUserError,
            error_details=f"Unknown authentication type: {str(authentication_type)}"
        )
    )


def _get_complete_additional_headers(
    model_type: ModelType,
    additional_headers: Optional[str],
    deployment_name: Optional[str],
) -> Dict[str, str]:
    if additional_headers:
        try:
            additional_headers_dict = json.loads(additional_headers)
        except json.JSONDecodeError as err:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details=f"additional_headers provided is not a valid \
                        stringified json string. Error: {str(err)}"
                )
            )
        if not isinstance(additional_headers_dict, dict):
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details="additional_headers provided is not a valid dictionary of key value pairs."
                )
            )
    else:
        additional_headers_dict = {}
    if deployment_name and model_type is not ModelType.OAI:
        additional_headers_dict["azureml-model-deployment"] = deployment_name
    return additional_headers_dict


@swallow_all_exceptions(logger)
def main(
        scoring_url: str,
        connection_name: str,
        authentication_type: AuthenticationType,
        debug_mode: bool,
        ensure_ascii: bool,
        initial_worker_count: int,
        max_worker_count: int,
        model_type: ModelType,
        response_segment_size: int,
        batch_score_config_path: str,
        additional_headers: Optional[str],
        deployment_name: Optional[str],
        max_retry_time_interval: Optional[int],
) -> None:
    """
    Entry script for the script.

    :param scoring_url: The URL of the endpoint.
    :param connection_name: The name of the connection to fetch the API_KEY for the
        endpoint authentication.
    :param authentication_type: Authentication type for endpoint.
    :param debug_mode: Enable debug mode will print all the debug logs in the score step.
    :param ensure_ascii: If true, the output will have all non-ASCII characters escaped.
    :param initial_worker_count: The initial number of workers to use for scoring.
    :param max_worker_count: Overrides initial_worker_count if necessary
    :param model_type: The type of the model.
    :param response_segment_size: The maximum number of tokens to generate at a time.
    :param batch_score_config_path: The config json file for the batch score component.
    :param additional_headers: A stringified json expressing additional headers to be added
        to each request.
    :param deployment_name: The deployment name. Only needed for managed OSS deployment.
    :param max_retry_time_interval: The maximum time (in seconds) spent retrying a payload.
        If unspecified, payloads are retried unlimited times.
    :return: None
    """

    endpoint_type = "azure_openai" if model_type is ModelType.OAI else "azureml_online_endpoint"
    authentication_dict = _get_authentication_config(authentication_type, connection_name)
    additional_headers_dict = _get_complete_additional_headers(
        model_type=model_type,
        additional_headers=additional_headers,
        deployment_name=deployment_name
    )

    config_dict = {
        "api": {
            "type": "completion", # TODO: verify this for chat scenario.
            "response_segment_size": response_segment_size
        },
        "authentication": authentication_dict,
        "concurrency_settings": {
            "initial_worker_count": initial_worker_count,
            "max_worker_count": max_worker_count,
        },
        "inference_endpoint": {
            "type": endpoint_type,
            "url": scoring_url,
        },
        "output_settings": {
            "save_partitioned_scoring_results": False, # TODO: try this with both true and false.
            "ensure_ascii": ensure_ascii,
        },
        "request_settings": {
            "headers": additional_headers_dict,
            "timeout": max_retry_time_interval,
        },
        "log_settings": {
            "stdout_log_level": "debug" if debug_mode else "info"
        }
    }
    logger.info("Saving config file")
    save_json_to_file(config_dict, batch_score_config_path)
    logger.info("Saved config file.")


if __name__ == "__main__":
    args = parse_args()
    main(
        scoring_url=args.scoring_url,
        connection_name=args.connection_name,
        authentication_type=AuthenticationType(args.authentication_type),
        debug_mode=args.debug_mode,
        ensure_ascii=args.ensure_ascii,
        initial_worker_count=args.initial_worker_count,
        max_worker_count=args.max_worker_count,
        batch_score_config_path=args.batch_score_config_path,
        additional_headers=args.additional_headers,
        deployment_name=args.deployment_name,
        max_retry_time_interval=args.max_retry_time_interval,
        model_type=ModelType(args.model_type),
        response_segment_size=args.response_segment_size,
    )
