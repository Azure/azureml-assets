# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing utility functions to dynamically setup the environment."""

import os
from pathlib import Path
from typing import List

from azureml.acft.common_components import get_logger_app

from utils.constants import EnvironmentConstants


logger = get_logger_app(__name__)


def save_list_to_txt(list_str: List[str], save_txt_path: str):
    """Save the list to a text file."""
    with open(save_txt_path, 'w', encoding="utf-8") as wptr:
        wptr.write("\n".join(list_str))
    logger.info(f"List saved to {save_txt_path}")


def set_up_environment(uninstalls: List[str], installs: List[str]):
    """Dynamic uninstallation and installing of user specified packages."""

    # Create the temp folder to save the installs and uninstalls
    Path(EnvironmentConstants.ENVIRONMENT_TMP_FOLDER).mkdir(exist_ok=True)

    # Uninstall the packages
    if uninstalls:
        logger.info(f"Uninstalling the following packages: {uninstalls}")
        save_list_to_txt(uninstalls, EnvironmentConstants.UNINSTALL_REQUIREMENTS_PATH)
        # TODO Replace os.system with subprocess.call
        # TODO Check if poetry can do smart uninstalls / installs
        os.system(f"pip uninstall -r {EnvironmentConstants.UNINSTALL_REQUIREMENTS_PATH} -y")

    # Install the packages
    if installs:
        logger.info(f"Installing the following packages: {installs}")
        save_list_to_txt(installs, EnvironmentConstants.REQUIREMENTS_PATH)
        os.system(f"pip install -r {EnvironmentConstants.REQUIREMENTS_PATH}")

    # Print the final package list
    os.system("pip list")
