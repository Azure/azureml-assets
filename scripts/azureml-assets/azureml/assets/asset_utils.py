# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tools to maintain asset source files."""

import argparse
import shutil
from pathlib import Path

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

ASSET_COUNT = "asset_count"
DELETED_COUNT = "deleted_count"


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

    NOTE: This is generally meant to be run against the release branch, where there's
    no risk of deleting shared files like component source code.

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
            # Get common directory, in case asset config file isn't at the root
            common_dir, _ = util.find_common_directory(asset_config.release_paths)

            # Delete asset
            if args.dry_run:
                logger.print(f"Would delete {asset_config.partial_name} from {common_dir}")
            else:
                try:
                    logger.print(f"Deleting {asset_config.partial_name} from {common_dir}")
                    shutil.rmtree(common_dir)
                    deleted_count += 1
                except Exception as e:
                    logger.log_warning(f"Failed to delete {common_dir}: {e}")

    # Set variables
    logger.set_output(ASSET_COUNT, asset_count)
    logger.set_output(DELETED_COUNT, deleted_count)


if __name__ == '__main__':
    # Quick function to convert comma-separated arg to Path list
    def _list_path(value: str):
        return [Path(d) for d in value.split(",")]

    # Handle command-line args
    parser = argparse.ArgumentParser()
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument("-i", "--input-dirs", type=_list_path, required=True,
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
