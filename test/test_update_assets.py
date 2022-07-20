# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import pytest
import shutil
import tempfile
from git import Repo
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util

RESOURCES_DIR = Path("resources/update")


@pytest.mark.parametrize(
    "test_subdir,create_tag",
    [
        ("in-subdir", True),
        ("in-parent-dir", False),
        ("manual-version", True),
        ("manual-version-unreleased", False),
    ]
)
def test_validate_assets(test_subdir: str, create_tag: bool):
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
            repo.create_tag(assets.get_release_tag_name(asset_config))

        # Create updatable expected dir
        if expected_dir.exists():
            shutil.copytree(expected_dir, temp_expected_path, dirs_exist_ok=True)
            expected_asset_config = util.find_assets(input_dirs=temp_expected_path)[0]
            assets.pin_env_files(expected_asset_config.environment_config_as_object())

        assets.update_assets(input_dirs=main_dir, asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
                             release_directory_root=temp_release_path, copy_only=False, skip_unreleased=False,
                             output_directory_root=temp_output_path)

        assert util.are_dir_trees_equal(temp_output_path, temp_expected_path, True)
