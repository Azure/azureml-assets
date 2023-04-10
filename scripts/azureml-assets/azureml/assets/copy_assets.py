# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Copy assets from directory to another."""

import argparse
import re
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

COPIED_COUNT = "copied_count"


def copy_asset(asset_config: assets.AssetConfig,
               output_directory_root: Path,
               release_directory_root: Path = None,
               use_version_dir: bool = False) -> str:
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

    # Copy asset to output directory
    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory_root, add_subdir=True,
                                  use_version_dir=use_version_dir)
    return asset_config.version


def copy_assets(input_dirs: List[Path],
                output_directory_root: Path,
                asset_config_filename: str,
                release_directory_root: Path = None,
                use_version_dirs: bool = False,
                pattern: re.Pattern = None):
    """Copy assets to output directory.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.
        output_directory_root (Path, optional): Output directory
        asset_config_filename (str): Asset config filename to search for.
        release_directory_root (Path, optional): Release directory location. Defaults to None.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
        pattern (re.Pattern, optional): Regex pattern for assets to copy. Defaults to None.
    """
    # Find assets under release dir
    asset_count = 0
    copied_count = 0
    for asset_config in util.find_assets(input_dirs, asset_config_filename, pattern=pattern):
        asset_count += 1

        # Copy asset if tag doesn't exist or release_directory_root isn't specified
        version = copy_asset(asset_config=asset_config,
                             output_directory_root=output_directory_root,
                             release_directory_root=release_directory_root,
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
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Copy assets
    copy_assets(input_dirs=input_dirs,
                output_directory_root=args.output_directory,
                asset_config_filename=args.asset_config_filename,
                release_directory_root=args.release_directory,
                use_version_dirs=args.use_version_dirs,
                pattern=args.pattern)
