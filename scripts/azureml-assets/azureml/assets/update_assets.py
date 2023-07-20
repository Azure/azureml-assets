# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Update assets to prepare for release."""

import argparse
import shutil
import tempfile
from collections import Counter
from git import Repo
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.environment as environment
import azureml.assets.util as util
from azureml.assets.util import logger

ASSET_COUNT = "asset_count"
UPDATED_COUNT = "updated_count"
UPDATED_ENV_COUNT = "updated_env_count"
COUNTERS = [ASSET_COUNT, UPDATED_COUNT, UPDATED_ENV_COUNT]


def pin_env_files(env_config: assets.EnvironmentConfig):
    """Pin versions in environment files marked as templates.

    Args:
        env_config (assets.EnvironmentConfig): Environment config

    Raises:
        Exception: Any failures to pin
    """
    files_to_pin = env_config.template_files_with_path

    # Replace template tags in environment config
    if assets.Config._contains_template(env_config.image_name):
        files_to_pin.append(env_config.file_name_with_path)

    # Replace template tags in files to pin
    for file_to_pin in files_to_pin:
        if file_to_pin.exists():
            try:
                environment.transform_file(file_to_pin)
            except Exception as e:
                raise Exception(f"Failed to pin versions in {file_to_pin}: {e}")
        else:
            logger.log_warning(f"Failed to pin versions in {file_to_pin}: File not found")


def release_tag_exists(asset_config: assets.AssetConfig, release_directory_root: Path) -> bool:
    """Check repo for an asset's release tag.

    Args:
        asset_config (assets.AssetConfig): Asset config
        release_directory_root (Path): Release branch location

    Returns:
        bool: True if the tag exists, False otherwise
    """
    # Check git repo for version-specific tag
    repo = Repo(release_directory_root)
    return asset_config.full_name in repo.tags


def get_latest_release_tag_version(asset_config: assets.AssetConfig, release_directory_root: Path) -> str:
    """Check repo to see if an asset's previous version was released if a latest tag exists.

    Args:
        asset_config (assets.AssetConfig): Asset config
        release_directory_root (Path): Release branch location

    Returns:
        str: Latest version found, or None if asset is not an environment or no tags not found
    """
    repo = Repo(release_directory_root)
    tags = [t for t in repo.tags if t.name.startswith(f"{asset_config.partial_name}/")]

    if not tags:
        # No releases
        return None

    # Get the latest tag
    ordered_tags = sorted(tags, key=lambda t: t.commit.authored_datetime)
    latest_tag = ordered_tags[-1].name
    _, _, latest_version = assets.AssetConfig.parse_full_name(latest_tag)

    return latest_version


def _update_asset_files(asset_config: assets.AssetConfig):
    """Update asset files.

    Args:
        asset_config (assets.AssetConfig): Asset config
    """
    if asset_config.type == assets.AssetType.ENVIRONMENT:
        env_config = asset_config.extra_config_as_object()
        pin_env_files(env_config)


def update_asset(asset_config: assets.AssetConfig,
                 output_directory_root: Path = None,
                 release_directory_root: Path = None,
                 skip_unreleased: bool = False,
                 use_version_dir: bool = False) -> str:
    """Update asset to prepare for release.

    Args:
        asset_config (assets.AssetConfig): Asset config
        output_directory_root (Path, optional): Output directory for updated assets. Defaults to None.
        release_directory_root (Path, optional): Release branch location. Defaults to None.
        skip_unreleased (bool, optional): Skip unreleased explicitly-versioned assets. Defaults to False.
        use_version_dir (bool, optional): Use version directory for output. Defaults to False.

    Returns:
        str: Version of updated asset, or None if not updated
    """
    # The release directory is required if auto-versioning
    if asset_config.auto_version and release_directory_root is None:
        logger.log_error(f"Asset {asset_config.name} is auto-versioned but can't be updated because no release "
                         "directory was specified to compare against")
        exit(1)

    # To keep things simple, we'll create a temporary directory for each update
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        # Copy asset to temp directory and pin image/package versions
        if output_directory_root is not None or release_directory_root is not None:
            temp_asset_dir = util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=temp_dir_path,
                                                           add_subdir=True)
            temp_asset_config = util.find_assets(input_dirs=temp_dir_path,
                                                 asset_config_filename=asset_config.file_name)[0]
            _update_asset_files(temp_asset_config)

        # Get version info, set a few defaults
        main_version = asset_config.version
        auto_version = asset_config.auto_version
        release_version = None
        pending_release = False

        # Look in release directory for existing asset
        if release_directory_root is not None:
            release_dir = util.get_asset_release_dir(asset_config, release_directory_root)
            if release_dir.exists():
                # Check existing release dir
                release_asset_configs = util.find_assets(input_dirs=release_dir,
                                                         asset_config_filename=asset_config.file_name)
                if not release_asset_configs:
                    logger.log_error(f"Release directory {release_dir} exists, but it's missing an asset config file")
                    exit(1)
                release_asset_config = release_asset_configs[0]
                release_version = release_asset_config.version

                # Update spec file
                assets.update_spec(temp_asset_config, version=release_version)

                # Compare temporary version with one in release
                dirs_equal = util.are_dir_trees_equal(temp_asset_dir, release_dir)
                if dirs_equal:
                    return None

                # See if the asset version is unreleased
                pending_release = not release_tag_exists(release_asset_config, release_directory_root)
                if pending_release and not auto_version and main_version != release_version and skip_unreleased:
                    # Skip the unreleased asset version
                    logger.log_warning(f"Skipping {release_asset_config.type.value} {release_asset_config.name} "
                                       f"because version {release_version} hasn't been released yet")
                    return None

        # Determine new version
        if not auto_version:
            # Use explicit version
            new_version = main_version
        elif pending_release:
            # Reuse existing auto version
            new_version = release_version
        else:
            # Increment auto version
            new_version = str(int(release_version) + 1 if release_version else 1)

        # Just update in place if no output directory
        if output_directory_root is None:
            _update_asset_files(asset_config)
            assets.update_spec(asset_config, version=new_version)
            return new_version

        # Identify asset's output directory
        output_is_release = (release_directory_root is not None and output_directory_root is not None
                             and output_directory_root.exists()
                             and output_directory_root.samefile(release_directory_root))
        if output_is_release:
            # Prevent release directory corruption
            use_version_dir = False
        output_directory = util.get_asset_output_dir(asset_config, output_directory_root, use_version_dir)

        # Copy and replace any existing directory
        util.copy_replace_dir(source=temp_asset_dir, dest=output_directory)

        # Update version in spec by copying clean spec and updating it
        asset_config_relative_path = temp_asset_config.file_name_with_path.relative_to(temp_asset_dir)
        output_asset_config = assets.AssetConfig(output_directory / asset_config_relative_path)
        shutil.copyfile(asset_config.spec_with_path, output_asset_config.spec_with_path)
        assets.update_spec(output_asset_config, version=new_version)

        return new_version


def update_assets(input_dirs: List[Path],
                  asset_config_filename: str,
                  output_directory_root: Path = None,
                  release_directory_root: Path = None,
                  skip_unreleased: bool = False,
                  use_version_dirs: bool = False):
    """Update assets to prepare for release.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        asset_config_filename (str): Asset config filename to search for
        output_directory_root (Path, optional): Output directory for updated assets. Defaults to None.
        release_directory_root (Path, optional): Release directory location. Defaults to None.
        skip_unreleased (bool, optional): Skip unreleased explicitly-versioned assets. Defaults to False.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
    """
    # Find assets under input dirs
    counters = Counter()
    for asset_config in util.find_assets(input_dirs, asset_config_filename):
        counters[ASSET_COUNT] += 1

        # Update asset if it's changed
        new_version = update_asset(asset_config=asset_config,
                                   output_directory_root=output_directory_root,
                                   release_directory_root=release_directory_root,
                                   skip_unreleased=skip_unreleased,
                                   use_version_dir=use_version_dirs)
        if new_version:
            logger.print(f"Updated {asset_config.type.value} {asset_config.name} version {new_version}")
            counters[UPDATED_COUNT] += 1

            # Track updated environments
            if asset_config.type == assets.AssetType.ENVIRONMENT:
                counters[UPDATED_ENV_COUNT] += 1
        else:
            logger.log_debug(f"No changes detected for {asset_config.type.value} {asset_config.name}")
    logger.print(f"{counters[UPDATED_COUNT]} of {counters[ASSET_COUNT]} asset(s) updated")

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", type=Path,
                        help="Copy new/updated assets into this directory, can be the same as --release-directory "
                        "and if omitted, assets will be updated in place")
    parser.add_argument("-r", "--release-directory", type=Path,
                        help="Directory to which the release branch has been cloned; if specified, only unreleased "
                        "and updated assets will be copied to the output directory")
    parser.add_argument("-s", "--skip-unreleased", action="store_true",
                        help="Skip unreleased explicitly-versioned assets in the release branch")
    parser.add_argument("-v", "--use-version-dirs", action="store_true",
                        help="Use version directories when storing assets in output directory")
    args = parser.parse_args()

    # Check interdependencies
    if args.skip_unreleased and args.release_directory is None:
        parser.error("--skip-unreleased requires --release-directory")
    if args.use_version_dirs and args.output_directory is None:
        parser.error("--use-version-dirs requires --output-directory")

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Update assets
    update_assets(input_dirs=input_dirs,
                  asset_config_filename=args.asset_config_filename,
                  output_directory_root=args.output_directory,
                  release_directory_root=args.release_directory,
                  skip_unreleased=args.skip_unreleased,
                  use_version_dirs=args.use_version_dirs)
