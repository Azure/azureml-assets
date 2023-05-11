# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tools to maintain asset source files."""

import argparse
import re
import shutil
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

ASSET_COUNT = "asset_count"
DELETED_COUNT = "deleted_count"


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
    logger.set_output(DELETED_COUNT, copied_count)


def list_assets(args: argparse.Namespace):
    """List assets.

    Args:
        args (argparse.Namespace): Args from argparse.
    """
    # Find assets under input dir
    asset_list = []
    for asset_config in util.find_assets(args.input_dirs, args.asset_config_filename):
        asset_list.append(asset_config.partial_name)
    asset_list.sort()

    # Write to file or stdout
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write("\n".join(asset_list))
    else:
        print("\n".join(asset_list))

    # Set variables
    logger.set_output(ASSET_COUNT, len(asset_list))


def delete_assets(args: argparse.Namespace):
    """Delete assets that are not in the retention file.

    Args:
        args (argparse.Namespace): Args from argparse.
    """
    # Read retention file
    with open(args.retention_file) as f:
        retention_list = f.read().splitlines()
        print(f"Read {len(retention_list)} asset(s) from retention file")

    # Find assets under input dir
    asset_count = 0
    deleted_count = 0
    for asset_config in util.find_assets(args.input_dirs, args.asset_config_filename):
        asset_count += 1
        if asset_config.partial_name not in retention_list:
            # Delete asset
            if args.dry_run:
                logger.print(f"Would delete {asset_config.partial_name}")
            else:
                try:
                    logger.print(f"Deleting {asset_config.partial_name}")
                    shutil.rmtree(asset_config.file_path)
                    deleted_count += 1
                except Exception as e:
                    logger.log_warning(f"Failed to delete {asset_config.partial_name}: {e}")

    # Set variables
    logger.set_output(ASSET_COUNT, asset_count)
    logger.set_output(DELETED_COUNT, deleted_count)


if __name__ == '__main__':
    # Quick function to convert comma-separated arg to Path list
    def list_path(value: str):
        return [Path(d) for d in value.split(",")]

    # Handle command-line args
    parser = argparse.ArgumentParser()
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("-i", "--input-dirs", type=list_path, required=True,
                               help="Comma-separated list of directories containing assets")
    shared_parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                               help="Asset config file name to search for")
    subparsers = parser.add_subparsers()

    parser_list = subparsers.add_parser("list", help="List assets", parents=[shared_parser])
    parser_list.set_defaults(func=list_assets)
    parser_list.add_argument("-o", "--output-file", type=Path,
                             help="File to which asset names will be written")

    parser_delete = subparsers.add_parser("delete", help="Delete assets", parents=[shared_parser])
    parser_delete.set_defaults(func=delete_assets)
    parser_delete.add_argument("-r", "--retention-file", required=True, type=Path,
                               help="File containing names of assets that will not be deleted")
    parser_delete.add_argument("-d", "--dry-run", action="store_true",
                               help="Dry run, don't actually make changes")
    args = parser.parse_args()
    args.func(args)
