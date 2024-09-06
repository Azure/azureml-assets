# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test util class."""

from pathlib import Path

import azureml.assets as assets
from azureml.assets.util.util import resolve_from_file_for_asset, is_file_relative_to_asset_path

RESOURCES_DIR = Path("resources/config")


def test_is_file_relative_to_asset_path():
    """Test provided path is a file under asset path."""
    this_dir = Path(__file__).parent
    resources = this_dir / RESOURCES_DIR

    asset_config = assets.AssetConfig(resources / "model-tagfiles" / assets.DEFAULT_ASSET_FILENAME)

    is_file = is_file_relative_to_asset_path(asset_config, "notes.md")
    assert is_file is True

    is_file = is_file_relative_to_asset_path(asset_config, Path("notes.md"))
    assert is_file is True

    is_file = is_file_relative_to_asset_path(asset_config, "doesnotexsit.md")
    assert is_file is False

    is_file = is_file_relative_to_asset_path(asset_config, Path("doesnotexsit.md"))
    assert is_file is False

    is_file = is_file_relative_to_asset_path(asset_config, 1.0)
    assert is_file is False

    is_file = is_file_relative_to_asset_path(asset_config, asset_config._append_to_file_path(Path("notes.md")))
    assert is_file is True

    is_file = is_file_relative_to_asset_path(asset_config, asset_config._append_to_file_path("notes.md"))
    assert is_file is True

    is_file = is_file_relative_to_asset_path(asset_config, asset_config._append_to_file_path(Path("doesnotexsit.md")))
    assert is_file is False


def test_resolve_from_file_for_asset():
    """Test resolving file content from file under asset path."""
    this_dir = Path(__file__).parent
    resources = this_dir / RESOURCES_DIR

    asset_config = assets.AssetConfig(resources / "model-tagfiles" / assets.DEFAULT_ASSET_FILENAME)

    content = resolve_from_file_for_asset(asset_config, "notes.md")
    assert content == "These are the notes!"

    content = resolve_from_file_for_asset(asset_config, Path("notes.md"))
    assert content == "These are the notes!"

    content = resolve_from_file_for_asset(asset_config, "doesnotexist.md")
    assert content == "doesnotexist.md"

    content = resolve_from_file_for_asset(asset_config, Path("doesnotexist.md"))
    assert content == Path("doesnotexist.md")

    content = resolve_from_file_for_asset(asset_config, 1.0)
    assert content == 1.0

    content = resolve_from_file_for_asset(asset_config, asset_config._append_to_file_path(Path("notes.md")))
    assert content == "These are the notes!"

    content = resolve_from_file_for_asset(asset_config, asset_config._append_to_file_path("notes.md"))
    assert content == "These are the notes!"

    content = resolve_from_file_for_asset(asset_config, asset_config._append_to_file_path(Path("doesnotexist.md")))
    assert content == asset_config._append_to_file_path(Path("doesnotexist.md"))
