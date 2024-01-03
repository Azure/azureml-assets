# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration parser factory."""

import sys

from .configuration_parser import ConfigurationParser
from .file_configuration_parser import FileConfigurationParser
from .file_configuration_validator import FileConfigurationValidator


class ConfigurationParserFactory:
    """Factory to create configuration parser based on the configuration mode."""

    def get_parser(self, args: 'list[str]' = None):
        """Create a parser based on the job arguments."""
        if args is None:
            args = sys.argv

        if '--configuration_file' in args:
            return FileConfigurationParser(FileConfigurationValidator())

        return ConfigurationParser()
