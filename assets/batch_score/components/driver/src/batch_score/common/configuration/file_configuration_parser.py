# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File configuration parser."""

import json
from argparse import ArgumentParser

from .command_line_argument_specification import (
    COMMAND_LINE_ARGUMENT_SPECIFICATION_FOR_FILE_CONFIGURATION,
)
from .configuration import Configuration
from .file_configuration_validator import FileConfigurationValidator


class FileConfigurationParser:
    """Parser for file-base configuration."""

    def __init__(self, validator: FileConfigurationValidator) -> None:
        """Init function."""
        self._validator = validator

    def parse_configuration(self, args: "list[str]" = None) -> Configuration:
        """Parse configuration."""
        parsed_args, _ = self._setup_parser().parse_known_args(args=args)
        config = self._validator.validate(parsed_args.configuration_file)

        async_mode = parsed_args.async_mode

        additional_headers = config.get('request_settings', {}).get('headers')
        if isinstance(additional_headers, dict):
            additional_headers = json.dumps(additional_headers)

        additional_properties = config.get('request_settings', {}).get('properties')
        if isinstance(additional_properties, dict):
            additional_properties = json.dumps(additional_properties)

        if config.get('output_settings', {}).get('save_partitioned_scoring_results'):
            output_behavior = 'summary_only'
            save_mini_batch_results = "enabled"
        else:
            output_behavior = 'append_row'
            save_mini_batch_results = "disabled"

        configuration = Configuration(
            additional_headers=additional_headers,
            additional_properties=additional_properties,
            api_key_name=None,
            api_type=config.get('api', {}).get('type'),
            app_insights_connection_string=config.get('log_settings', {}).get('app_insights_connection_string'),
            async_mode=async_mode,
            authentication_type=config.get('authentication', {}).get('type'),
            batch_pool=None,
            batch_size_per_request=config.get('api', {}).get('batch_size_per_request'),
            configuration_file=None,
            connection_name=config.get('authentication', {}).get('name'),
            debug_mode=config.get('log_settings', {}).get('app_insights_log_level') == 'debug',
            ensure_ascii=config.get('output_settings', {}).get('ensure_ascii'),
            image_input_folder=None,
            initial_worker_count=config.get('concurrency_settings', {}).get('initial_worker_count'),
            max_retry_time_interval=config.get('request_settings', {}).get('timeout'),
            max_worker_count=config.get('concurrency_settings', {}).get('max_worker_count'),
            mini_batch_results_out_directory=parsed_args.partitioned_scoring_results,
            online_endpoint_url=None,
            output_behavior=output_behavior,
            quota_audience=None,
            quota_estimator=None,
            request_path=None,
            save_mini_batch_results=save_mini_batch_results,
            scoring_url=config.get('inference_endpoint', {}).get('url'),
            segment_large_requests=config.get('api', {}).get('response_segment_size') > 0,
            segment_max_token_size=config.get('api', {}).get('response_segment_size'),
            service_namespace=None,
            tally_exclusions=None,
            tally_failed_requests=None,
            token_file_path=None,
            user_agent_segment=config.get('request_settings', {}).get('user_agent_identifier'),
        )

        return configuration

    def _setup_parser(self) -> ArgumentParser:
        parser = ArgumentParser()

        for argument, options in COMMAND_LINE_ARGUMENT_SPECIFICATION_FOR_FILE_CONFIGURATION.items():
            parser.add_argument(argument, **options)

        return parser
