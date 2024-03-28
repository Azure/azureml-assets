# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component Configurations."""

from argparse import Namespace
from dataclasses import dataclass, field

from .. import constants
from ..common_enums import EndpointType, ApiType, AuthenticationType
from ..telemetry import logging_utils as lu


@dataclass(frozen=True)
class Configuration(Namespace):
    """Component configurations."""

    additional_headers: str = field(init=True, default=None)
    additional_properties: str = field(init=True, default=None)
    api_key_name: str = field(init=True, default=None)
    api_type: str = field(init=True, default=None)
    app_insights_connection_string: str = field(init=True, default=None)
    app_insights_log_level: str = field(init=True, default="debug")
    async_mode: bool = field(init=True, default=False)
    authentication_type: str = field(init=True, default=None)
    batch_pool: str = field(init=True, default=None)
    batch_size_per_request: int = field(init=True, default=1)
    configuration_file: str = field(init=True, default=None)
    connection_name: str = field(init=True, default=None)
    debug_mode: bool = field(init=True, default=None)
    ensure_ascii: bool = field(init=True, default=None)
    image_input_folder: str = field(init=True, default=None)
    initial_worker_count: int = field(init=True, default=None)
    input_schema_version: int = field(init=True, default=1)
    max_retry_time_interval: int = field(init=True, default=None)
    max_worker_count: int = field(init=True, default=None)
    mini_batch_results_out_directory: str = field(init=True, default=None)
    online_endpoint_url: str = field(init=True, default=None)
    output_behavior: str = field(init=True, default=None)
    quota_audience: str = field(init=True, default=None)
    quota_estimator: str = field(init=True, default=None)
    request_path: str = field(init=True, default=None)
    save_mini_batch_results: str = field(init=True, default=None)
    scoring_url: str = field(init=True, default=None)
    segment_large_requests: str = field(init=True, default=None)
    segment_max_token_size: int = field(init=True, default=None)
    service_namespace: str = field(init=True, default=None)
    split_output: bool = field(init=True, default=False)
    stdout_log_level: str = field(init=True, default="debug")
    tally_exclusions: str = field(init=True, default=None)
    tally_failed_requests: bool = field(init=True, default=None)
    token_file_path: str = field(init=True, default=None)
    user_agent_segment: str = field(init=True, default=None)

    def __post_init__(self) -> None:
        """Post init function."""
        self._validate()

    def _validate(self):
        self._validate_batch_size_per_request()
        self._validate_online_endpoint_url_and_request_path()
        self._validate_segment_large_requests()

    def _validate_batch_size_per_request(self):
        if self.batch_size_per_request < 1:
            raise ValueError("The optional parameter 'batch_size_per_request' cannot be less than 1."
                             " Valid range is 1-2000.")

        if self.batch_size_per_request > 2000:
            raise ValueError("The optional parameter 'batch_size_per_request' cannot be greater than 2000."
                             " Valid range is 1-2000.")

        if self.batch_size_per_request > 1 and not self.is_embeddings():
            raise ValueError("The optional parameter 'batch_size_per_request' is only allowed to be "
                             "greater than 1 for the Embeddings API. Valid range is 1-2000.")

    def _validate_online_endpoint_url_and_request_path(self):
        if (self.online_endpoint_url
                and self.request_path is not None
                and self.request_path not in constants.DEFAULT_REQUEST_PATHS):
            raise ValueError("The optional parameter 'online_endpoint_url' is not allowed in combination "
                             "with 'request_path'. Please put the entire scoring url in the "
                             "`online_endpoint_url` parameter and remove 'request_path'.")

    def _validate_segment_large_requests(self):
        if self.segment_large_requests == 'enabled' and not self.is_completion():
            raise ValueError("The optional parameter 'segment_large_requests' is supported only with "
                             "the Completion API. Please set 'segment_large_requests' to 'disabled' or "
                             "remove it from the configuration.")

    def log(self):
        """Log the configuration as alpha-sorted key-value pairs."""
        logger = lu.get_logger()
        for arg, value in sorted(vars(self).items()):
            if arg == "app_insights_connection_string" and value is not None:
                value = "[redacted]"

            logger.debug(f"{arg}: {value}")

    def is_embeddings(self) -> bool:
        """Check if the target endpoint is for embeddings models."""
        return (
            self.request_path == constants.DV_EMBEDDINGS_API_PATH
            or (
                self.scoring_url is not None
                and (
                    self.scoring_url.endswith(constants.DV_EMBEDDINGS_API_PATH)
                    or self.api_type in [ApiType.Embedding, "embeddings"]
                )
            )
        )

    def is_chat_completion(self) -> bool:
        """Check if the target endpoint is for chat completion models."""
        return (
            self.request_path == constants.DV_CHAT_COMPLETIONS_API_PATH
            or (
                self.scoring_url is not None
                and (
                    self.scoring_url.endswith(constants.DV_CHAT_COMPLETIONS_API_PATH)
                    or self.api_type == ApiType.ChatCompletion
                )
            )
        )

    def is_completion(self) -> bool:
        """Check if the target endpoint is for completion models."""
        return (
            self.request_path == constants.DV_COMPLETION_API_PATH
            or (
                self.scoring_url is not None
                and (
                    self.scoring_url.endswith(constants.DV_COMPLETION_API_PATH)
                    or self.api_type == ApiType.Completion
                )
            )
        )

    def is_sahara(self) -> bool:
        """Check if the target endpoint is for sahara models."""
        return self.batch_pool and self.batch_pool.lower() == "sahara-global"

    def is_vesta(self) -> bool:
        """Check if the target endpoint is for vesta models."""
        return (
            self.request_path == constants.VESTA_RAINBOW_API_PATH
            or (
                self.scoring_url is not None
                and (
                    self.scoring_url.endswith(constants.VESTA_RAINBOW_API_PATH)
                    or self.api_type == ApiType.Vesta
                )
            )
        )

    def is_vesta_chat_completion(self) -> bool:
        """Check if the target endpoint is for vesta chat completion models."""
        return (
            self.request_path == constants.VESTA_CHAT_COMPLETIONS_API_PATH
            or (
                self.scoring_url is not None
                and (
                    self.scoring_url.endswith(constants.VESTA_CHAT_COMPLETIONS_API_PATH)
                    or self.api_type == ApiType.VestaChatCompletion
                )
            )
        )

    def is_aoai_endpoint(self) -> bool:
        """Check if the target endpoint is for Azure OpenAI models."""
        return self.scoring_url and \
            any(suffix in self.scoring_url for suffix in constants.AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST)

    def is_serverless_endpoint(self) -> bool:
        """Check if the target endpoint is MIR serverless."""
        return self.scoring_url and constants.SERVERLESS_ENDPOINT_DOMAIN_SUFFIX in self.scoring_url

    def get_endpoint_type(self) -> EndpointType:
        """Get endpoint type."""
        if self.is_aoai_endpoint():
            return EndpointType.AOAI
        elif self.is_serverless_endpoint():
            return EndpointType.Serverless
        elif self.batch_pool and self.quota_audience and self.service_namespace:
            return EndpointType.BatchPool
        else:
            return EndpointType.MIR

    def get_api_type(self) -> ApiType:
        """Get api type."""
        if self.is_completion():
            return ApiType.Completion
        elif self.is_chat_completion():
            return ApiType.ChatCompletion
        elif self.is_vesta():
            return ApiType.Vesta
        elif self.is_vesta_chat_completion():
            return ApiType.VestaChatCompletion
        elif self.is_embeddings():
            return ApiType.Embedding
        else:
            return ApiType.Unknown

    def get_authentication_type(self) -> AuthenticationType:
        """Get authentication type."""
        if self.authentication_type == AuthenticationType.ApiKey:
            return AuthenticationType.ApiKey
        elif self.authentication_type == AuthenticationType.ManagedIdentity:
            return AuthenticationType.ManagedIdentity
        elif self.authentication_type in [constants.CONNECTION_AUTH_TYPE, AuthenticationType.WorkspaceConnection]:
            return AuthenticationType.WorkspaceConnection
        else:
            return AuthenticationType.Unknown
