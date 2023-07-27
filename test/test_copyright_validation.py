# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test copyright_validation script."""

from pathlib import Path
import pytest
import subprocess

VALIDATION_SCRIPT = Path("scripts/validation/copyright_validation.py")
RESOURCES_DIR = Path("resources/validate-copyright")


@pytest.mark.parametrize(
    "test_subdir,expected",
    [
        ("good-validation", True),
        ("missing-copyright", False),
    ]
)
def test_copyright_validation(test_subdir: str, expected: bool):
    """Test validate_assets function.

    Args:
        test_subdir (str): Test subdirectory
        expected (bool): Success expected
    """
    this_dir = Path(__file__).parent
    validation_script = this_dir.parent / VALIDATION_SCRIPT

    input_dir = this_dir / RESOURCES_DIR / test_subdir
    result = subprocess.run(["python", validation_script, "-i", input_dir])
    assert (result.returncode == 0) == expected
