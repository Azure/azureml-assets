# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Copy assets from directory to another."""

import argparse
import re
from collections import Counter
from pathlib import Path
from string import Template
from typing import List
from urllib.error import HTTPError

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger
from azureml.assets.config import AssetType

COPIED_COUNT = "copied_count"
COPIED_TYPE_COUNT = Template("copied_${type}_count")


def copy_asset(asset_config: assets.AssetConfig,
               output_directory_root: Path,
               release_directory_root: Path = None,
               use_version_dir: bool = False,
               check_previous_release: bool = False) -> str:
    """Copy asset to output directory.

    Args:
        asset_config (assets.AssetConfig): Asset config.
        output_directory_root (Path, optional): Output directory.
        release_directory_root (Path, optional): Release directory location. Defaults to None.
        use_version_dir (bool, optional): Use version directory for output. Defaults to False.
    """
    if release_directory_root is not None and assets.release_tag_exists(asset_config, release_directory_root):
        # Skip a released version
        return None

    if check_previous_release and asset_config.type == AssetType.ENVIRONMENT:
        # Skip if previous version was not released
        previous_release_version = assets.get_latest_release_tag_version(asset_config, release_directory_root)

        if previous_release_version is not None:
            image = asset_config.extra_config_as_object().get_full_image_name()

            # Check against MCR by making a manifest call to see if tag exists
            (hostname, repo) = image.split("/", 1)

            try:
                _ = assets.get_manifest(previous_release_version, hostname, repo)
            except HTTPError as e:
                if e.code == 404:
                    logger.log_error(f"Image {image} not found in MCR. Please release {asset_config.name} version "
                                     f"{previous_release_version} before continuing.")

                    exit(1)
                else:
                    logger.log_error(f"Unexpected error when looking for image {image} in MCR")
                    raise Exception(f"Failed to retrieve manifest for {repo}:{previous_release_version}: {e}")
            except Exception as e:
                raise Exception(f"Failed to retrieve manifest for {repo}:{previous_release_version}: {e}")

    # Copy asset to output directory
    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory_root, add_subdir=True,
                                  use_version_dir=use_version_dir)

    return asset_config.version


def copy_assets(input_dirs: List[Path],
                changed_files: List[Path],
                output_directory_root: Path,
                asset_config_filename: str,
                release_directory_root: Path = None,
                use_version_dirs: bool = False,
                pattern: re.Pattern = None,
                check_previous_release: bool = False):
    """Copy assets to output directory.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        changed_files (List[Path]): List of changed files
        output_directory_root (Path, optional): Output directory
        asset_config_filename (str): Asset config filename to search for.
        release_directory_root (Path, optional): Release directory location. Defaults to None.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    # Find assets under release dir
    asset_count = 0
    copied_count = 0
    copied_type_counter = Counter()
    for asset_config in util.find_assets(
            input_dirs, asset_config_filename, changed_files=changed_files, pattern=pattern):
        asset_count += 1

        # Copy asset if tag doesn't exist or release_directory_root isn't specified
        version = copy_asset(asset_config=asset_config,
                             output_directory_root=output_directory_root,
                             release_directory_root=release_directory_root,
                             use_version_dir=use_version_dirs,
                             check_previous_release=check_previous_release)
        if version:
            logger.print(f"Copied {asset_config.type.value} {asset_config.name} version {version}")
            copied_count += 1
            copied_type_counter[asset_config.type] += 1
    logger.print(f"{copied_count} of {asset_count} asset(s) copied")

    # Set variables
    logger.set_output(COPIED_COUNT, copied_count)
    for type, count in copied_type_counter.items():
        logger.set_output(COPIED_TYPE_COUNT.substitute(type=type.value), count)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")
    parser.add_argument("-o", "--output-directory", required=True, type=Path,
                        help="Directory to which assets will be written")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-r", "--release-directory", type=Path,
                        help="Directory to which the release branch has been cloned. "
                        "If provided, only asset versions without a release tag will be copied.")
    parser.add_argument("-v", "--use-version-dirs", action="store_true",
                        help="Use version directories when storing assets in output directory")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    parser.add_argument("-c", "--changed-files", type=str,
                        help="Comma-separated list of changed files, used to filter assets")
    parser.add_argument("-p", "--check-previous-release", action="store_true",
                        help="Check if previous release exists (environments only)")
    args = parser.parse_args()

    # Ensure --release-directory is specified if --check-previous-release is
    if args.check_previous_release and args.release_directory is None:
        parser.error("--check-previous-release requires --release-directory")

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Copy assets
    copy_assets(input_dirs=input_dirs,
                changed_files=changed_files,
                output_directory_root=args.output_directory,
                asset_config_filename=args.asset_config_filename,
                release_directory_root=args.release_directory,
                use_version_dirs=args.use_version_dirs,
                pattern=args.pattern,
                check_previous_release=args.check_previous_release)
