# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate source tree files."""

import argparse
import sys
from pathlib import Path
from typing import List

import azureml.assets.util as util
from azureml.assets.util import logger
import os.path

BYTES_IN_A_MEGABYTE = 1024 ** 2


def validate_tree(input_dirs: List[Path]):
    """Validate source tree files.

    Args:
        input_dirs (List[Path]): List of directories to search for assets.

    Returns:
        bool: True if files were successfully validated, otherwise False.
    """
    error_count = 0

    # Check that every spec.yaml has asset.yaml next to it
    for spec_config_file in util.find_files(input_dirs, "spec.yaml"):
        asset_config_file = str(spec_config_file).replace("spec", "asset")
        if not os.path.exists(asset_config_file):
            logger.log_error(f"{spec_config_file} does not have a corresponding asset.yaml")
            error_count += 1

    # Fail if any asset.yml in the tree
    for asset_yml_file in util.find_files(input_dirs, "asset.yml"):
        logger.log_error(f"{asset_yml_file} should be named asset.yaml")
        error_count += 1

    # Fail if any spec.yml in the tree
    for spec_yml_file in util.find_files(input_dirs, "spec.yml"):
        logger.log_error(f"{spec_yml_file} should be named spec.yaml")
        error_count += 1

    # Scan every file in the source tree to be 1 MB or less
    for file in util.find_files(input_dirs, "*"):
        if os.path.getsize(file) > BYTES_IN_A_MEGABYTE:
            logger.log_error(f"{file} should not be over 1 MB")
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
