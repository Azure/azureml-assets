# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test validate_assets script."""

from pathlib import Path
import pytest

import azureml.assets as assets

RESOURCES_DIR = Path("resources/validate")


@pytest.mark.parametrize(
    "test_subdir,check_images,expected",
    [
        ("name-mismatch", False, False),
        ("version-mismatch", False, False),
        ("invalid-strings", False, False),
        ("env-with-underscores", False, False),
        ("framework-ver-missing", False, False),
        ("ubuntu-in-name", False, False),
        ("extra-gpu", False, False),
        ("incorrect-order", False, False),
        ("image-name-mismatch", True, False),
        ("publishing-disabled", True, False),
        ("good-validation", True, True),
        ("correct-order", True, True),
        ("missing-description-file", True, False),
        ("missing-copyright", False, False),
    ]
)
def test_validate_assets(test_subdir: str, check_images: bool, expected: bool):
    """Test validate_assets function.

    Args:
        test_subdir (str): Test subdirectory
        check_images (bool): Check image build/publish info
        expected (bool): Success expected
    """
    this_dir = Path(__file__).parent

    assert assets.validate_assets(
        input_dirs=this_dir / RESOURCES_DIR / test_subdir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
        check_images=check_images) == expected
