from argparse import Namespace
from dataclasses import dataclass, field

from ...batch_pool.routing.routing_client import RoutingClient
from ...batch_pool.scoring.scoring_client import ScoringClient
from .. import constants
from ..auth.auth_provider import EndpointType
from ..telemetry import logging_utils as lu


@dataclass(frozen=True)
class Configuration(Namespace):
    additional_headers: str = field(init=True, default=None)
    additional_properties: str = field(init=True, default=None)
    configuration_file: str = field(init=True, default=None)
    api_key_name: str = field(init=True, default=None)
    api_type: str = field(init=True, default=None)
    app_insights_connection_string: str = field(init=True, default=None)
    async_mode: bool = field(init=True, default=False)
    authentication_type: str = field(init=True, default=None)
    batch_pool: str = field(init=True, default=None)
    batch_size_per_request: int = field(init=True, default=1)
    connection_name: str = field(init=True, default=None)
    debug_mode: bool = field(init=True, default=None)
    ensure_ascii: bool = field(init=True, default=None)
    image_input_folder: str = field(init=True, default=None)
    initial_worker_count: int = field(init=True, default=None)
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
    tally_exclusions: str = field(init=True, default=None)
    tally_failed_requests: bool = field(init=True, default=None)
    token_file_path: str = field(init=True, default=None)
    user_agent_segment: str = field(init=True, default=None)

    def __post_init__(self) -> None:
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
            raise ValueError("The optional parameter 'batch_size_per_request' is only allowed to be greater than 1 for the Embeddings API."
                             " Valid range is 1-2000.")

    def _validate_online_endpoint_url_and_request_path(self):
        if (self.online_endpoint_url
            and self.request_path is not None
            and self.request_path not in ScoringClient.DEFAULT_REQUEST_PATHS
        ):
            raise ValueError("The optional parameter 'online_endpoint_url' is not allowed in combination with 'request_path'. "
                            "Please put the entire scoring url in the `online_endpoint_url` parameter and remove 'request_path'.")

    def _validate_segment_large_requests(self):
        if self.segment_large_requests == 'enabled' and not self.is_completion():
            raise ValueError("The optional parameter 'segment_large_requests' is supported only with the Completion API."
                             "Please set 'segment_large_requests' to 'disabled' or remove it from the configuration.")

    def log(self):
        """Log the configuration as alpha-sorted key-value pairs.
        """
        logger = lu.get_logger()
        for arg, value in sorted(vars(self).items()):
            if arg == "app_insights_connection_string" and value is not None:
                value = "[redacted]"

            logger.debug(f"{arg}: {value}")

    def is_embeddings(self) -> bool:
        return self.request_path == ScoringClient.DV_EMBEDDINGS_API_PATH or\
            (self.online_endpoint_url and self.online_endpoint_url.endswith(ScoringClient.DV_EMBEDDINGS_API_PATH)) or\
            (self.scoring_url and self.api_type == constants.EMBEDDINGS_API_TYPE)

    def is_chat_completion(self) -> bool:
        return self.request_path == ScoringClient.DV_CHAT_COMPLETIONS_API_PATH or\
            (self.online_endpoint_url and self.online_endpoint_url.endswith(ScoringClient.DV_CHAT_COMPLETIONS_API_PATH)) or\
            (self.scoring_url and self.api_type == constants.CHAT_COMPLETION_API_TYPE)

    def is_completion(self) -> bool:
        return self.request_path == ScoringClient.DV_COMPLETION_API_PATH or\
            (self.online_endpoint_url and self.online_endpoint_url.endswith(ScoringClient.DV_COMPLETION_API_PATH)) or\
            (self.scoring_url and self.api_type == constants.COMPLETION_API_TYPE)

    def is_sahara(self, routing_client: RoutingClient) -> bool:
        return routing_client and routing_client.target_batch_pool and routing_client.target_batch_pool.lower() == "sahara-global"

    def is_vesta(self) -> bool:
        return self.request_path == ScoringClient.VESTA_RAINBOW_API_PATH or\
            (self.online_endpoint_url and self.online_endpoint_url.endswith(ScoringClient.VESTA_RAINBOW_API_PATH)) or\
            (self.scoring_url and self.api_type == constants.VESTA_API_TYPE)

    def is_vesta_chat_completion(self) -> bool:
        return self.request_path == ScoringClient.VESTA_CHAT_COMPLETIONS_API_PATH or\
            (self.online_endpoint_url and self.online_endpoint_url.endswith(ScoringClient.VESTA_CHAT_COMPLETIONS_API_PATH)) or\
            (self.scoring_url and self.api_type == constants.VESTA_CHAT_COMPLETION_API_TYPE)
    
    def is_aoai_endpoint(self) -> bool:
        return self.scoring_url and any(suffix in self.scoring_url for suffix in constants.AOAI_ENDPOINT_DOMAIN_SUFFIX_LIST)
    
    def is_serverless_endpoint(self) -> bool:
        return self.scoring_url and constants.SERVERLESS_ENDPOINT_DOMAIN_SUFFIX in self.scoring_url

    def get_endpoint_type(self) -> EndpointType:
        if (self.is_aoai_endpoint()):
            return EndpointType.AOAI
        elif (self.is_serverless_endpoint()):
            return EndpointType.Serverless
        else:
            return EndpointType.MIR
