# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test update_assets script."""

import pytest
import shutil
import tempfile
from git import Repo
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util

RESOURCES_DIR = Path("resources/update")


@pytest.mark.parametrize(
    "test_subdir,skip_unreleased,create_tag",
    [
        ("in-subdir", False, True),
        ("in-parent-dir", False, False),
        ("manual-version", False, True),
        ("manual-version-unreleased", False, False),
        ("manual-version-unreleased-skip", True, False),
    ]
)
def test_update_assets(test_subdir: str, skip_unreleased: bool, create_tag: bool):
    """Test update_assets function.

    Args:
        test_subdir (str): Test subdirectory
        skip_unreleased (bool): Value to pass to update_assets
        create_tag (bool): Create release tag in temp repo
    """
    this_dir = Path(__file__).parent
    test_dir = this_dir / RESOURCES_DIR / test_subdir
    main_dir = test_dir / "main"
    release_dir = test_dir / "release"
    expected_dir = test_dir / "expected"

    # Temp directory helps keep the original release directory clean
    with tempfile.TemporaryDirectory(prefix="release-") as temp_dir1, \
            tempfile.TemporaryDirectory(prefix="output-") as temp_dir2, \
            tempfile.TemporaryDirectory(prefix="expected-") as temp_dir3:
        temp_release_path = Path(temp_dir1)
        temp_output_path = Path(temp_dir2)
        temp_expected_path = Path(temp_dir3)

        # Create fake release branch
        shutil.copytree(release_dir, temp_release_path, dirs_exist_ok=True)
        repo = Repo.init(temp_release_path)
        for path in (temp_release_path / "latest").rglob("*"):
            if path.is_dir():
                continue
            rel_path = path.relative_to(temp_release_path)
            repo.index.add(str(rel_path))
        repo.git.config("user.email", "<>")
        repo.git.config("user.name", "Unit Test")
        repo.git.commit("-m", "Initial commit")

        # Create tag
        if create_tag:
            asset_config = util.find_assets(input_dirs=temp_release_path)[0]
            repo.create_tag(asset_config.full_name)

        # Create updatable expected dir
        if expected_dir.exists():
            shutil.copytree(expected_dir, temp_expected_path, dirs_exist_ok=True)
            expected_asset_config = util.find_assets(input_dirs=temp_expected_path)[0]
            if expected_asset_config.type == assets.AssetType.ENVIRONMENT:
                assets.pin_env_files(expected_asset_config.extra_config_as_object())

        assets.update_assets(input_dirs=main_dir, asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
                             release_directory_root=temp_release_path, copy_only=False,
                             skip_unreleased=skip_unreleased, output_directory_root=temp_output_path)

        assert util.are_dir_trees_equal(temp_output_path, temp_expected_path, True)
