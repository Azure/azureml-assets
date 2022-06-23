import argparse
import shutil
import tempfile
from git import Repo
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.environment as environment
import azureml.assets.util as util
from azureml.assets.util import logger

RELEASE_TAG_VERSION_TEMPLATE = "{type}/{name}/{version}"
HAS_UPDATES = "has_updates"
ENV_OS_UPDATES = "env_os_updates"


def pin_env_files(env_config: assets.EnvironmentConfig):
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


def get_release_tag_name(asset_config: assets.AssetConfig):
    version = asset_config.spec_as_object().version
    return RELEASE_TAG_VERSION_TEMPLATE.format(type=asset_config.type.value, name=asset_config.name,
                                               version=version)


def release_tag_exists(asset_config: assets.AssetConfig, release_directory_root: Path) -> bool:
    # Check git repo for version-specific tag
    repo = Repo(release_directory_root)
    tag = get_release_tag_name(asset_config)
    return tag in repo.tags


def update_asset(asset_config: assets.AssetConfig,
                 release_directory_root: Path,
                 copy_only: bool,
                 skip_unreleased: bool,
                 output_directory_root: Path = None) -> str:
    # Determine asset's release directory
    release_dir = util.get_asset_release_dir(asset_config, release_directory_root)

    # Define output directory, which may be different from the release directory
    if output_directory_root:
        output_directory = util.get_asset_output_dir(asset_config, output_directory_root)
    else:
        output_directory = release_dir

    # Simpler operation that just copies the directory
    if copy_only:
        util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory)
        return asset_config.spec_as_object().version

    # Get version from main branch, set a few defaults
    main_version = asset_config.version
    auto_version = asset_config.auto_version
    release_version = None
    check_contents = False
    pending_release = False

    # Check existing release dir
    if release_dir.exists():
        release_asset_config = util.find_assets(input_dirs=release_dir,
                                                asset_config_filename=asset_config.file_name)[0]
        release_version = release_asset_config.version
        check_contents = True

        # See if the asset version is unreleased
        pending_release = not release_tag_exists(release_asset_config, release_directory_root)
        if pending_release and ((not auto_version and main_version != release_version) or skip_unreleased):
            # Skip the unreleased asset version
            logger.log_warning(f"Skipping {release_asset_config.type.value} {release_asset_config.name} because "
                               f"version {release_version} hasn't been released yet")
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
        new_version = int(release_version) + 1 if release_version else 1

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        # Copy asset to temp directory and pin image/package versions
        util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=temp_dir_path, add_subdir=True)
        temp_asset_config = util.find_assets(input_dirs=temp_dir_path, asset_config_filename=asset_config.file_name)[0]
        temp_env_config = temp_asset_config.environment_config_as_object()
        if temp_env_config:
            pin_env_files(temp_env_config)

        temp_asset_dir = util.get_asset_output_dir(asset_config=asset_config, output_directory_root=temp_dir_path)
        
        # Compare temporary version with one in release
        if check_contents:
            assets.update_spec(temp_asset_config, version=release_version)
            dirs_equal = util.are_dir_trees_equal(temp_asset_dir, release_dir, enable_logging=True)
            if dirs_equal:
                return None

        # Copy and replace any existing directory
        util.copy_replace_dir(source=temp_asset_dir, dest=output_directory)

        # Update version in spec by copying clean spec and updating it
        asset_config_relative_path = temp_asset_config.file_name_with_path.relative_to(temp_asset_dir)
        output_asset_config = assets.AssetConfig(output_directory / asset_config_relative_path)
        shutil.copyfile(asset_config.spec_with_path, output_asset_config.spec_with_path)
        assets.update_spec(output_asset_config, version=str(new_version))

        return new_version


def update_assets(input_dirs: List[Path],
                  asset_config_filename: str,
                  release_directory_root: Path,
                  copy_only: bool,
                  skip_unreleased: bool,
                  output_directory_root: Path = None):
    # Find assets under input dirs
    asset_count = 0
    updated_count = 0
    updated_os = set()
    for asset_config in util.find_assets(input_dirs, asset_config_filename):
        asset_count += 1

        # Update asset if it's changed
        new_version = update_asset(asset_config=asset_config,
                                   release_directory_root=release_directory_root,
                                   copy_only=copy_only,
                                   skip_unreleased=skip_unreleased,
                                   output_directory_root=output_directory_root)
        if new_version:
            logger.print(f"Updated {asset_config.type.value} {asset_config.name} version {new_version}")
            updated_count += 1

            # Track updated environments by OS
            if asset_config.type is assets.AssetType.ENVIRONMENT:
                temp_env_config = asset_config.environment_config_as_object()
                updated_os.add(temp_env_config.os.value)
        else:
            logger.log_debug(f"No changes detected for {asset_config.type.value} {asset_config.name}")
    logger.print(f"{updated_count} of {asset_count} asset(s) updated")

    # Set variables
    logger.set_output(HAS_UPDATES, "true" if updated_count > 0 else "false")
    logger.set_output(ENV_OS_UPDATES, ",".join(updated_os))


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing assets")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, type=Path, help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--output-directory", type=Path, help="Directory to which new/updated assets will be written, defaults to release directory")
    parser.add_argument("-c", "--copy-only", action="store_true", help="Just copy assets into the release directory")
    parser.add_argument("-s", "--skip-unreleased", action="store_true", help="Skip unreleased dynamically-versioned assets in the release branch")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Update assets
    update_assets(input_dirs=input_dirs,
                  asset_config_filename=args.asset_config_filename,
                  release_directory_root=args.release_directory,
                  copy_only=args.copy_only,
                  skip_unreleased=args.skip_unreleased,
                  output_directory_root=args.output_directory)
