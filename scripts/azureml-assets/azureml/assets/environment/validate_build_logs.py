# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate build log for additional vulnerabilities."""

import os
import argparse
import re
import sys
from pathlib import Path
from azureml.assets.util import logger


def validate_py_version(build_log_file_name, build_log_content):
    """Validate Python version.

    Args:
        build_log_file_name (str): Build log file name.
        build_log_content (str): Build log content

    Returns:
        int: Number of errors.
    """
    py38_match = re.search("python=3.8", build_log_content)

    if py38_match:
        logger.log_error(f"{build_log_file_name}: python=3.8 found in build log."
                         f"Python 3.8 is now deprecated. Please use a newer Python version.")
        return 1

    return 0


def validate_build_logs(build_logs_dir):
    """Validate environment build logs.

    Args:
        build_logs_dir (Path): Directory of environment build logs.

    Returns:
        bool: True if build logs were successfully validated, otherwise False.
    """
    error_count = 0

    for build_log_file_name in os.listdir(build_logs_dir):

        build_log_file_path = os.path.join(build_logs_dir, build_log_file_name)
        print(f"Validating {build_log_file_name} for additional vulnerabilities")

        with open(build_log_file_path, "r") as f:
            build_log_content = f.read()
            error_count += validate_py_version(build_log_file_name, build_log_content)

    return error_count == 0


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--build-logs-dir", required=True, type=Path,
                        help="Directory of build logs")
    args = parser.parse_args()

    if not os.path.isdir(args.build_logs_dir):
        print(f"No build logs found since {args.build_logs_dir} does not exist")
        sys.exit(0)

    # Validate build logs
    success = validate_build_logs(build_logs_dir=args.build_logs_dir)

    if success:
        print("No additional vulnerabilities found in build logs")
    else:
        sys.exit(1)
