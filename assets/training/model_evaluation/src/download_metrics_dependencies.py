# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script for downloading and installing runtime dependencies for compute metrics."""

from logging_utilities import (
    custom_dimensions, get_logger, swallow_all_exceptions
)
from azureml.telemetry.activity import log_activity
from constants import TelemetryConstants

import sys
import subprocess

custom_dimensions.app_name = TelemetryConstants.DOWNLOAD_METRICS_DEPENDENCIES
logger = get_logger(name=__name__)
custom_dims_dict = vars(custom_dimensions)


@swallow_all_exceptions(logger)
def main():
    """Download and Install compute metrics dependencies."""
    with log_activity(logger, TelemetryConstants.DOWNLOAD_METRICS_DEPENDENCIES,
                      custom_dimensions=custom_dims_dict):
        install_deps = [
            "azure-ai-ml",
        ]

        no_install = []
        if len(install_deps) > 0:
            try:
                command = [sys.executable, '-m', 'pip', 'install'] + install_deps
                command_str = ' '.join(command)
                logger.info(f"Installing dependencies. Executing command: {command_str}")
                subprocess.check_output(command, stderr=subprocess.STDOUT)
            except Exception as e:
                logger.warning(f"Installing dependencies failed with error:\n{e.stdout.decode('utf-8')}")
                logger.info("Installing dependencies one by one.")

                command = [sys.executable, '-m', 'pip', 'install', 'dep']
                for dep in install_deps:
                    try:
                        command[-1] = dep
                        command_str = ' '.join(command)
                        logger.info(f"Installing dependencies. Executing command: {command_str}")
                        subprocess.check_output(command, stderr=subprocess.STDOUT)
                    except Exception as e:
                        logger.warning(f"Installing dependency {dep} failed with error:\n{e.stdout.decode('utf-8')}")
                        no_install += [dep]

        if len(no_install):
            no_install = "\n".join(no_install)
            logger.warning(
                f"Could not install following dependencies - \n{no_install}"
                f"\nSome functionality might not work as expected."
            )


if __name__ == "__main__":
    main()
