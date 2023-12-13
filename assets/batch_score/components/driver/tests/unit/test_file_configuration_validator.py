import os
from pathlib import Path

import pytest

from src.batch_score.common.configuration.file_configuration_validator import (
    FileConfigurationValidator,
    InvalidConfigurationError,
    list_files_recursively,
)


configuration_files_dir = (
    Path(os.getcwd())
    / "driver"
    / "test_assets"
    / "configuration_files"
)

valid_dir = configuration_files_dir / "valid"
invalid_dir = configuration_files_dir / "invalid"


@pytest.mark.parametrize("instance", list_files_recursively(valid_dir))
def test_valid_configurations_pass(instance):
    FileConfigurationValidator().validate(instance)


@pytest.mark.parametrize("instance", list_files_recursively(invalid_dir))
@pytest.mark.xfail(strict=True, raises=InvalidConfigurationError)
def test_invalid_configurations_fail(instance):
    FileConfigurationValidator().validate(instance)
