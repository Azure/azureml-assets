# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from argparse import ArgumentParser

from .configuration import Configuration
from .command_line_argument_specification import COMMAND_LINE_ARGUMENT_SPECIFICATION


class ConfigurationParser:
    def parse_configuration(self, args: "list[str]" = None) -> Configuration:
        ''' Parses the command line arguments and returns a Configuration object.
            If args are provided, they are parsed. Otherwise, sys.argv is parsed.
        '''

        parsed_args, _ = self._setup_parser().parse_known_args(args=args)
        args_dict = vars(parsed_args)
        # Custom solution provided for Benchmark.
        args_dict = ConfigurationParser._update_configuration_from_file(args_dict)
        args_dict = ConfigurationParser._set_defaults(args_dict)

        return Configuration(**args_dict)

    @staticmethod
    def _update_configuration_from_file(args: dict) -> dict:
        # Read from JSON file and override parameter values
        if args['configuration_file'] is None:
            return args

        with open(args['configuration_file'], 'r') as json_file:
            configuration = json.load(json_file)

        # TODO: Override all parameter values specified in the file
        for arg in ["scoring_url", "authentication_type", "connection_name"]:
            if arg in configuration and configuration.get(arg) is not None:
                args[arg] = configuration[arg]

        return args

    @staticmethod
    def _set_defaults(args: dict) -> dict:
        '''
        The v1.5, v2 and v2_singularity components expose `online_endpoint_url` whereas LLM component
        exposes `scoring_url`. With the assignment below, all of the internal logic will use `scoring_url`.
        '''
        if args.get('online_endpoint_url'):
            args['scoring_url'] = args['online_endpoint_url']

        # if segment_large_requests is not set, then default it.
        # set to enabled if completion api, disabled otherwise
        if args.get('segment_large_requests') is None:
            if Configuration(**args).is_completion():
                args['segment_large_requests'] = 'enabled'
            else:
                args['segment_large_requests'] = 'disabled'

        # Override log levels for backward compactability
        if args.get('debug_mode') is None or args.get('debug_mode'):
            args['app_insights_log_level'] = 'debug'
            args['stdout_log_level'] = 'debug'
        else:
            args['app_insights_log_level'] = 'info'
            args['stdout_log_level'] = 'info'

        return args

    def _setup_parser(self) -> ArgumentParser:
        parser = ArgumentParser()

        for argument, options in COMMAND_LINE_ARGUMENT_SPECIFICATION.items():
            parser.add_argument(argument, **options)

        return parser
