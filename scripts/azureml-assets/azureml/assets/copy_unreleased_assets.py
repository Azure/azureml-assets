# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Copy unreleased assets from release directory to output directory."""

import argparse
import re
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

COPIED_COUNT = "copied_count"


def copy_unreleased_asset(asset_config: assets.AssetConfig,
                          release_directory_root: Path,
                          output_directory_root: Path,
                          use_version_dir: bool = False) -> str:
    """Copy unreleased asset from release directory to output directory.

    Args:
        asset_config (assets.AssetConfig): Asset config.
        release_directory_root (Path): Release directory location.
        output_directory_root (Path, optional): Output directory.
        use_version_dir (bool, optional): Use version directory for output. Defaults to False.
    """
    if assets.release_tag_exists(asset_config, release_directory_root):
        # Skip a released version
        return None

    # Copy asset to output directory
    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory_root, add_subdir=True,
                                  use_version_dir=use_version_dir)
    return asset_config.spec_as_object().version


def copy_unreleased_assets(release_directory_root: Path,
                           output_directory_root: Path,
                           asset_config_filename: str,
                           use_version_dirs: bool = False,
                           pattern: re.Pattern = None):
    """Copy unreleased assets from release directory to output directory.

    Args:
        release_directory_root (Path): Release directory location.
        output_directory_root (Path, optional): Output directory
        asset_config_filename (str): Asset config filename to search for.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    # Find assets under release dir
    asset_count = 0
    copied_count = 0
    for asset_config in util.find_assets(release_directory_root, asset_config_filename, pattern=pattern):
        asset_count += 1

        # Copy asset if tag doesn't exist
        version = copy_unreleased_asset(asset_config=asset_config,
                                        release_directory_root=release_directory_root,
                                        output_directory_root=output_directory_root,
                                        use_version_dir=use_version_dirs)
        if version:
            logger.print(f"Copied {asset_config.type.value} {asset_config.name} version {version}")
            copied_count += 1
    logger.print(f"{copied_count} of {asset_count} asset(s) copied")

    # Set variables
    logger.set_output(COPIED_COUNT, copied_count)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--release-directory", required=True, type=Path,
                        help="Directory to which the release branch has been cloned")
    parser.add_argument("-o", "--output-directory", required=True, type=Path,
                        help="Directory to which unreleased assets will be written")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-v", "--use-version-dirs", action="store_true",
                        help="Use version directories when storing assets in output directory")
    parser.add_argument("-t", "--pattern", type=re.compile,
                        help="Regex pattern to select assets to copy, in the format <type>/<name>/<version>")
    args = parser.parse_args()

    # Release assets
    copy_unreleased_assets(release_directory_root=args.release_directory,
                           output_directory_root=args.output_directory,
                           asset_config_filename=args.asset_config_filename,
                           use_version_dirs=args.use_version_dirs,
                           pattern=args.pattern)
