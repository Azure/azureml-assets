# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom Engine module.

This module contains the CustomEngine class which is responsible for initializing the custom server,
generating responses for given prompts, and managing the server processes.
"""

import subprocess
import sys
import os
from foundation.model.serve.constants import EnvironmentVariables
from foundation.model.serve.engine.engine import BaseEngine
from foundation.model.serve.logging_config import configure_logger

logger = configure_logger(__name__)


class CustomEngine(BaseEngine):
    """Custom Engine class.

    This class is responsible for initializing the custom server, generating responses for given prompts,
    and managing the server processes.
    """

    def __init__(self):
        """Initialize the CustomEngine class with startup script path from environment."""
        self.startup_script_path = os.getenv(
            EnvironmentVariables.ENGINE_STARTUP_FILE_PATH, None)

    def init_server(self):
        """Initialize client[s] for the engine to receive requests on.

        Launches the custom engine by executing the startup script specified
        in the ENGINE_STARTUP_FILE_PATH environment variable.
        """
        logger.info("Starting custom engine with startup script.")
        if self.startup_script_path:
            subprocess.Popen([sys.executable, self.startup_script_path])
