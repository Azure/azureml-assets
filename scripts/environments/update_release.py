import argparse
import os
import pygit2
import shutil
import tempfile
from typing import List

from build import pin_env_files
from ci_logger import logger
from config import AssetConfig, AssetType, EnvironmentConfig, Spec
from update_spec import update as update_spec
from util import are_dir_trees_equal

TAG_TEMPLATE = "refs/tags/{name}"
RELEASE_TAG_VERSION_TEMPLATE = "{type}/{name}/{version}"
RELEASE_SUBDIR_TEMPLATE = "latest/{type}/{name}"


def release_tag_exists(asset_config: AssetConfig, release_directory_root: str):
    # Check git repo for version-specific tag
    repo = pygit2.Repository(release_directory_root)
    version_tag = RELEASE_TAG_VERSION_TEMPLATE.format(type=asset_config.type.value, name=asset_config.name,
                                                      version=asset_config.version)
    return repo.references.get(TAG_TEMPLATE.format(name=version_tag)) is not None


def update_asset(asset_config: AssetConfig, release_directory_root: str) -> str:
    # Determine asset's release directory
    release_subdir = RELEASE_SUBDIR_TEMPLATE.format(type=asset_config.type.value,
                                                    name=asset_config.name)
    release_dir = os.path.join(release_directory_root, release_subdir)

    # Get version from main branch, set a few defaults
    main_version = asset_config.version
    release_version = None
    check_contents = False

    # Check existing release dir
    if os.path.exists(release_dir):
        release_asset_config = AssetConfig(os.path.join(release_dir, asset_config.file_name))

        if main_version:
            # Explicit releases, just check version
            release_version = release_asset_config.version
            if main_version == release_version:
                # No version change
                return None
        else:
            # Dynamic releases, will need to check contents
            release_spec = Spec(release_asset_config.spec_with_path)
            release_version = release_spec.version
            check_contents = True

        if not release_tag_exists(release_asset_config, release_directory_root):
            # Skip a non-released version
            # TODO: Determine whether this should fail the workflow
            logger.log_warning(f"Skipping {release_asset_config.type.value} {release_asset_config.name} because "
                                f"version {release_version} hasn't been released yet")
            return None

    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy asset to temp directory and pin image/package versions
        shutil.copytree(asset_config.file_path, temp_dir, dirs_exist_ok=True)
        temp_asset_config = AssetConfig(os.path.join(temp_dir, asset_config.file_name))
        if asset_config.type is AssetType.ENVIRONMENT:
            temp_env_config = EnvironmentConfig(temp_asset_config.extra_config_with_path)
            pin_env_files(temp_env_config)

        # Compare temporary version with one in release
        if check_contents:
            update_spec(temp_asset_config, version=release_version)
            dirs_equal = are_dir_trees_equal(temp_dir, release_dir)
            if dirs_equal:
                return None

        # Delete release dir
        if os.path.exists(release_dir):
            shutil.rmtree(release_dir)

        # Copy temp asset to release dir
        shutil.copytree(temp_dir, release_dir)

        # Determine new version
        if main_version:
            # Explicit versioning
            new_version = main_version
        else:
            # Dynamic versioning
            new_version = int(release_version) + 1 if release_version else 1

        # Update version in spec by copying clean spec and updating it
        shutil.copyfile(asset_config.spec_with_path, os.path.join(release_dir, asset_config.spec))
        release_asset_config = AssetConfig(os.path.join(release_dir, asset_config.file_name))
        update_spec(release_asset_config, version=str(new_version))

        return new_version


def update_release(image_dirs: List[str], asset_config_filename: str, release_directory_root: str):
    # Find environments under image root directories
    asset_count = 0
    updated_count = 0
    for image_dir in image_dirs:
        for root, _, files in os.walk(image_dir):
            for asset_config_file in [f for f in files if f == asset_config_filename]:
                # Load config
                asset_config = AssetConfig(os.path.join(root, asset_config_file))
                asset_count += 1

                # Update asset if it's changed
                new_version = update_asset(asset_config=asset_config,
                                           release_directory_root=release_directory_root)
                if new_version:
                    print(f"Updated {asset_config.type.value} {asset_config.name} to version {new_version}")
                    updated_count += 1
                else:
                    logger.log_debug(f"No changes detected for {asset_config.type.value} {asset_config.name}")
    print(f"{updated_count} of {asset_count} asset(s) updated")


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dirs", required=True, help="Comma-separated list of directories containing image to build")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, help="Directory to which the release branch has been cloned")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_dirs = args.image_dirs.split(",")

    # Build images
    update_release(image_dirs=image_dirs, asset_config_filename=args.asset_config_filename,
                   release_directory_root=args.release_directory)
