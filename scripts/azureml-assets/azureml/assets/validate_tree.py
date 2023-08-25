# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate source tree files."""

import argparse
import sys
from pathlib import Path
from typing import List

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

BYTES_IN_A_MEGABYTE = 1024 ** 2


def validate_tree(input_dirs: List[Path]):
    """Validate source tree files.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.

    Returns:
        bool: True if files were successfully validated, otherwise False.
    """
    error_count = 0

    for file in util.find_files(input_dirs, "*"):
        if file.name == "spec.yaml":
            # Check that every spec.yaml has asset.yaml next to it
            asset_config_file = file.parent / assets.DEFAULT_ASSET_FILENAME
            if not asset_config_file.exists():
                logger.log_error(f"{file} does not have a corresponding {assets.DEFAULT_ASSET_FILENAME}")
                error_count += 1
        elif file.name == "asset.yml":
            # Fail if any asset.yml in the tree
            logger.log_error(f"{file} should be named {assets.DEFAULT_ASSET_FILENAME}")
            error_count += 1

        # Scan every file in the source tree to be 1 MB or less
        if file.stat().st_size > BYTES_IN_A_MEGABYTE:
            logger.log_error(f"{file} is too large (over 1MB)")
            error_count += 1

    return error_count == 0


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing assets")

    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Validate tree
    success = validate_tree(input_dirs=input_dirs)

    if not success:
        sys.exit(1)
