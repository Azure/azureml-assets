# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch benchmark config generator component."""

import argparse
import json

from typing import Any, Dict, Optional, Union

from aml_benchmark.utils.io import save_json_to_file
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.exceptions import swallow_all_exceptions
from aml_benchmark.utils.aml_run_utils import str2bool
from aml_benchmark.utils.exceptions import BenchmarkUserException
from aml_benchmark.utils.constants import AuthenticationType, ApiType, get_api_type, get_endpoint_type
from aml_benchmark.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the args for the method."""
    # Input and output arguments
    parser = argparse.ArgumentParser(description=f"{__file__}")
    parser.add_argument(
        "--configuration_file",
        type=str,
        required=False,
        default=None,
        help="An optional config file path that contains deployment configurations.",
    )
    parser.add_argument(
        "--scoring_url",
        type=str,
        help="The URL of the endpoint.",
        default=None,
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
        type=AuthenticationType,
        help="Authentication type for endpoint.",
        choices=list(AuthenticationType)
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
        help="The maximum time (in seconds) spent retrying a payload. \
            If unspecified, payloads are retried unlimited times.",
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
    parser.add_argument(
        "--app_insights_connection_string",
        type=str,
        required=False,
        help="The azure application insights connection string where metrics and logs will be logged."
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
                    error_details=(
                        "Connection name should be provided when authentication_type is"
                        " set to azureml_workspace_connection."
                    )
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
    endpoint_type: str,
    additional_headers: Optional[Union[str, Dict[Any, Any]]],
    deployment_name: Optional[str],
) -> Dict[str, str]:
    """
    Get the complete additional headers for the requests.

    :param endpoint_type: The type of the model endpoint.
    :param additional_headers: Additional headers for the model.
    :param deployment_name: The deployment name of the endpoint.
    :returns: The complete additional headers dictionary.
    """
    if additional_headers and isinstance(additional_headers, str):
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
                    error_details="additional_headers provided is not a valid dictionary \
                        of key value pairs."
                )
            )
    elif isinstance(additional_headers, dict):
        additional_headers_dict = additional_headers
    else:
        additional_headers_dict = {}
    if deployment_name and endpoint_type != "azure_openai":
        additional_headers_dict["azureml-model-deployment"] = deployment_name
    return additional_headers_dict


def _get_request_settings(
    additional_headers_dict: Dict[str, Any],
    max_retry_time_interval: Optional[int],
) -> Dict[str, Any]:
    """
    Get request settings config.

    :param additional_headers_dict: Additional headers for the requests.
    :param max_return_time_interval: The maximum time (in seconds) spent retrying a payload.
    :returns: The request settings config.
    """
    if not max_retry_time_interval:
        max_retry_time_interval = 0
    return {
        "headers": additional_headers_dict,
        "timeout": max_retry_time_interval
    }


def _get_overriding_configs(configuration_file: Optional[str]) -> Dict[Any, Any]:
    """
    Get overriding config file.

    :param configuration_file: Additional configuration file.
    :returns: The additional configuration dictionary.
    """
    config = {}
    if configuration_file:
        try:
            with open(configuration_file) as file:
                config_from_file = json.load(file)
        except Exception as ex:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details=f"Failed to read configuration file due to error: {str(ex)}"
                )
            )
        logger.info(f"Config from file: {config_from_file}")
        if isinstance(config_from_file, dict):
            config = config_from_file
        else:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details="configuration_file provided is not a valid dictionary of key value pairs."
                )
            )
    return config


@swallow_all_exceptions(logger)
def main(
    scoring_url: Optional[str],
    connection_name: str,
    authentication_type: AuthenticationType,
    debug_mode: bool,
    ensure_ascii: bool,
    initial_worker_count: int,
    max_worker_count: int,
    response_segment_size: int,
    batch_score_config_path: str,
    configuration_file: Optional[str],
    additional_headers: Optional[str],
    deployment_name: Optional[str],
    max_retry_time_interval: Optional[int],
    app_insights_connection_string: Optional[str],
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
    :param response_segment_size: The maximum number of tokens to generate at a time.
    :param batch_score_config_path: The config json file for the batch score component.
    :param configuration_file: An optional config file path that contains deployment configurations.
    :param additional_headers: A stringified json expressing additional headers to be added
        to each request.
    :param deployment_name: The deployment name. Only needed for managed OSS deployment.
    :param max_retry_time_interval: The maximum time (in seconds) spent retrying a payload.
        If unspecified, payloads are retried unlimited times.
    :param app_insights_connection_string: The application insights connection string.
    :return: None
    """
    # Override the configs if config file is passed. Config file takes precedence
    # as resource manager determined them based on deployment it did on fly.
    overriding_configs = _get_overriding_configs(configuration_file)
    logger.info(f"Overriding config is {overriding_configs}")
    merged_scoring_url = overriding_configs.get("scoring_url", scoring_url)
    merged_online_headers = overriding_configs.get("scoring_headers", additional_headers)
    merged_connection_name = overriding_configs.get("connection_name", connection_name)
    logger.info(f"scoring URL: {scoring_url}")
    logger.info(f"headers: {merged_online_headers}")
    logger.info(f"Connection name: {merged_connection_name}")

    endpoint_type = get_endpoint_type(merged_scoring_url)
    authentication_dict = _get_authentication_config(authentication_type, merged_connection_name)
    additional_headers_dict = _get_complete_additional_headers(
        endpoint_type=endpoint_type,
        additional_headers=merged_online_headers,
        deployment_name=deployment_name
    )
    request_settings_dict = _get_request_settings(additional_headers_dict, max_retry_time_interval)

    api_type = get_api_type(merged_scoring_url)
    api_dict = {"type": api_type}
    if api_type == ApiType.Completion:
        api_dict["response_segment_size"] = response_segment_size

    config_dict = {
        "api": api_dict,
        "authentication": authentication_dict,
        "concurrency_settings": {
            "initial_worker_count": initial_worker_count,
            "max_worker_count": max_worker_count,
        },
        "inference_endpoint": {
            "type": endpoint_type,
            "url": merged_scoring_url,
        },
        "output_settings": {
            "save_partitioned_scoring_results": True,
            "ensure_ascii": ensure_ascii,
        },
        "request_settings": request_settings_dict,
        "log_settings": {
            "stdout_log_level": "debug" if debug_mode else "info",
        }
    }
    if app_insights_connection_string:
        config_dict["log_settings"]["app_insights_connection_string"] = app_insights_connection_string
    logger.info("Saving config file")
    save_json_to_file(config_dict, batch_score_config_path)
    logger.info("Saved config file.")


if __name__ == "__main__":
    args = parse_args()
    main(
        configuration_file=args.configuration_file,
        scoring_url=args.scoring_url,
        connection_name=args.connection_name,
        authentication_type=args.authentication_type,
        debug_mode=args.debug_mode,
        ensure_ascii=args.ensure_ascii,
        initial_worker_count=args.initial_worker_count,
        max_worker_count=args.max_worker_count,
        batch_score_config_path=args.batch_score_config_path,
        additional_headers=args.additional_headers,
        deployment_name=args.deployment_name,
        max_retry_time_interval=args.max_retry_time_interval,
        response_segment_size=args.response_segment_size,
        app_insights_connection_string=args.app_insights_connection_string,
    )
