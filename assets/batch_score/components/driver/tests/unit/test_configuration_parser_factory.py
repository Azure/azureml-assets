# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for configuration parser factory."""

import pytest

from src.batch_score.common.configuration.configuration_parser import (
    ConfigurationParser,
)
from src.batch_score.common.configuration.configuration_parser_factory import (
    ConfigurationParserFactory,
)
from src.batch_score.common.configuration.file_configuration_parser import (
    FileConfigurationParser,
)


@pytest.mark.parametrize(
    "args, expected_parser_type",
    [
        (["--configuration_file", "path/to/configuration/file"], FileConfigurationParser),
        # typo in argument name, hyphen instead of underscore
        (["--configuration-file", "path/to/configuration/file"], ConfigurationParser),
        (["--other_param", "some value"], ConfigurationParser),
    ],
)
def test_factory_returns_parser(args, expected_parser_type):
    parser = ConfigurationParserFactory().get_parser(args=args)
    assert isinstance(parser, expected_parser_type)
