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
        (["--configuration-file", "path/to/configuration/file"], ConfigurationParser), # typo in argument name, hyphen instead of underscore
        (["--other_param", "some value"], ConfigurationParser),
    ],
)
def test_factory_returns_parser(args, expected_parser_type):
    parser = ConfigurationParserFactory().get_parser(args=args)
    assert isinstance(parser, expected_parser_type)
