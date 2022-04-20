import argparse
import os
import pygit2
import shutil
import tempfile
from typing import List

from build import pin_env_files
from ci_logger import logger
from collections import Counter
from config import AssetConfig, AssetType, EnvironmentConfig, Os, Spec
from update_spec import update as update_spec
from util import are_dir_trees_equal

TAG_TEMPLATE = "refs/tags/{name}"
RELEASE_TAG_VERSION_TEMPLATE = "{type}/{name}/{version}"
RELEASE_SUBDIR_TEMPLATE = "latest/{type}/{name}"
OUTPUT_SUBDIR_TEMPLATE = "{type}/{name}"
HAS_UPDATES = "has_updates"
OS_UPDATES = "os_updates"


def release_tag_exists(asset_config: AssetConfig, release_directory_root: str):
    # Check git repo for version-specific tag
    repo = pygit2.Repository(release_directory_root)
    version_tag = RELEASE_TAG_VERSION_TEMPLATE.format(type=asset_config.type.value, name=asset_config.name,
                                                      version=asset_config.version)
    return repo.references.get(TAG_TEMPLATE.format(name=version_tag)) is not None


def copy_replace_dir(source: str, dest: str):
    # Delete output dir
    if os.path.exists(dest):
        shutil.rmtree(dest)

    # Copy temp asset to output dir
    shutil.copytree(source, dest)


def update_asset(asset_config: AssetConfig,
                 release_directory_root: str,
                 copy_only: bool,
                 env_os_counter: Counter,
                 output_directory_root: str = None) -> str:
    # Determine asset's release directory
    release_subdir = RELEASE_SUBDIR_TEMPLATE.format(type=asset_config.type.value,
                                                    name=asset_config.name)
    release_dir = os.path.join(release_directory_root, release_subdir)

    # Define output directory, which may be different from the release directory
    if output_directory_root:
        output_subdir = OUTPUT_SUBDIR_TEMPLATE.format(type=asset_config.type.value,
                                                      name=asset_config.name)
        output_directory = os.path.join(output_directory_root, output_subdir)
    else:
        output_directory = release_dir

    # Simpler operation that just copies the directory
    if copy_only:
        copy_replace_dir(asset_config.file_path, output_directory)
        spec = Spec(asset_config.spec_with_path)
        return spec.version

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

        # Copy and replace any existing directory
        copy_replace_dir(temp_dir, output_directory)

        # Determine new version
        if main_version:
            # Explicit versioning
            new_version = main_version
        else:
            # Dynamic versioning
            new_version = int(release_version) + 1 if release_version else 1

        # Update version in spec by copying clean spec and updating it
        shutil.copyfile(asset_config.spec_with_path, os.path.join(output_directory, asset_config.spec))
        output_asset_config = AssetConfig(os.path.join(output_directory, asset_config.file_name))
        update_spec(output_asset_config, version=str(new_version))

        # Count by OS
        if asset_config.type is AssetType.ENVIRONMENT:
            temp_env_config = EnvironmentConfig(temp_asset_config.extra_config_with_path)
            env_os_counter[temp_env_config.os.value] += 1

        return new_version


def update_release(image_dirs: List[str],
                   asset_config_filename: str,
                   release_directory_root: str,
                   copy_only: bool,
                   output_directory_root: str = None):
    # Find environments under image root directories
    asset_count = 0
    updated_count = 0
    env_os_counter = Counter({i.value: 0 for i in list(Os)})
    for image_dir in image_dirs:
        for root, _, files in os.walk(image_dir):
            for asset_config_file in [f for f in files if f == asset_config_filename]:
                # Load config
                asset_config = AssetConfig(os.path.join(root, asset_config_file))
                asset_count += 1

                # Update asset if it's changed
                new_version = update_asset(asset_config=asset_config,
                                           release_directory_root=release_directory_root,
                                           copy_only=copy_only,
                                           env_os_counter=env_os_counter,
                                           output_directory_root=output_directory_root)
                if new_version:
                    print(f"Updated {asset_config.type.value} {asset_config.name} to version {new_version}")
                    updated_count += 1
                else:
                    logger.log_debug(f"No changes detected for {asset_config.type.value} {asset_config.name}")
    print(f"{updated_count} of {asset_count} asset(s) updated")

    # Set variables
    logger.set_output(HAS_UPDATES, "true" if updated_count > 0 else "false")
    logger.set_output(OS_UPDATES, ",".join([n for n, c in env_os_counter.items() if c > 0]))


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dirs", required=True, help="Comma-separated list of directories containing image to build")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", required=True, help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--output-directory", help="Directory to which new/updated assets will be written, defaults to release directory")
    parser.add_argument("-c", "--copy-only", action="store_true", help="Just copy assets into the release directory")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_dirs = args.image_dirs.split(",")

    # Build images
    update_release(image_dirs=image_dirs,
                   asset_config_filename=args.asset_config_filename,
                   release_directory_root=args.release_directory,
                   copy_only=args.copy_only,
                   output_directory_root=args.output_directory)
