# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test azureml.assets.test_assets() function."""

import os
import pytest
import subprocess
from pathlib import Path

RESOURCES_DIR = Path("resources/pytest")
SCRIPTS_DIR = Path("../scripts/test")
TEST_ASSETS_SCRIPT = SCRIPTS_DIR / "test_assets.py"
TEST_REQUIREMENTS_FILE = SCRIPTS_DIR / "requirements.txt"


@pytest.mark.parametrize(
    "test_subdir,expected",
    [
        ("good-assets-with-conda-environment", True),
        ("good-assets-with-requirements", True),
        ("bad-assets", False),
        ("mixed-assets", False)
    ]
)
def test_test_assets(test_subdir: str, expected: bool):
    """Test azureml.assets.test_assets() function."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    assert subscription_id is not None
    assert resource_group is not None
    assert workspace_name is not None

    completed_process = subprocess.run([
        "python",
        this_dir / TEST_ASSETS_SCRIPT,
        "-i", this_dir / RESOURCES_DIR / test_subdir,
        "-p", this_dir / TEST_REQUIREMENTS_FILE,
    ])
    assert (completed_process.returncode == 0) == expected
