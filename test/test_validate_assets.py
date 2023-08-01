# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test validate_assets script."""

from pathlib import Path
import pytest
import re

import azureml.assets as assets

RESOURCES_DIR = Path("resources/validate")


@pytest.mark.parametrize(
    "test_subdir,check_images,check_names,check_names_skip_pattern,check_dockerfile,expected",
    [
        ("name-mismatch", False, True, None, False, False),
        ("version-mismatch", False, True, None, False, False),
        ("invalid-strings", False, True, None, False, False),
        ("env-with-underscores", False, True, None, False, False),
        ("framework-ver-missing", False, True, None, False, False),
        ("ubuntu-in-name", False, True, None, False, False),
        ("ubuntu-in-name", False, True, re.compile(r"environment/env-ubuntu20.04/.+"), False, True),
        ("ubuntu-in-name", False, False, None, False, True),
        ("extra-gpu", False, None, True, False, False),
        ("incorrect-order", False, True, None, False, False),
        ("image-name-mismatch", True, True, None, False, False),
        ("publishing-disabled", True, True, None, False, False),
        ("good-validation", True, True, None, True, True),
        ("correct-order", True, True, None, True, False),
        ("missing-description-file", True, True, None, False, False),
        ("data-good", False, True, None, False, True),
        ("data-path-mismatch-1", False, True, None, False, False),
        ("data-path-mismatch-2", False, True, None, False, False),
        ("dockerfile-from-ce-image", False, False, None, True, False),
    ]
)
def test_validate_assets(test_subdir: str, check_images: bool, check_names: bool,
                         check_names_skip_pattern: re.Pattern, expected: bool):
    """Test validate_assets function.

    Args:
        test_subdir (str): Test subdirectory
        check_images (bool): Check image build/publish info
        check_names (bool): Check name
        check_names_skip_pattern (re.Pattern): Skip pattern for name validation
        expected (bool): Success expected
    """
    this_dir = Path(__file__).parent

    assert assets.validate_assets(
        input_dirs=this_dir / RESOURCES_DIR / test_subdir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
        check_names=check_names,
        check_names_skip_pattern=check_names_skip_pattern,
        check_images=check_images) == expected
