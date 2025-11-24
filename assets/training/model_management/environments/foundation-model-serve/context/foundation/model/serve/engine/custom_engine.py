# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom Engine module.

This module contains the CustomEngine class which is responsible for initializing the custom server,
generating responses for given prompts, and managing the server processes.
"""

import subprocess
import sys
import os
from typing import Optional
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
        super().__init__()
        self.startup_script_path = os.getenv(
            EnvironmentVariables.ENGINE_STARTUP_FILE_PATH, None)
        self.server_process: Optional[subprocess.Popen] = None

    def init_server(self):
        """Initialize client[s] for the engine to receive requests on.

        Launches the custom engine by executing the startup script specified
        in the ENGINE_STARTUP_FILE_PATH environment variable.
        """
        logger.info("Starting custom engine with startup script.")
        if self.startup_script_path:
            self.server_process = subprocess.Popen([sys.executable, self.startup_script_path])
            logger.info(f"Custom engine started with PID: {self.server_process.pid}")
        else:
            raise EnvironmentError(
                f"{EnvironmentVariables.ENGINE_STARTUP_FILE_PATH} environment variable is not set.")

    def restart_server(self):
        """Restart the custom engine server by stopping current process and starting new one.

        Raises:
            RuntimeError: If startup script path is not configured.
        """
        if not self.startup_script_path:
            raise RuntimeError("Startup script path not configured. Cannot restart server.")
        
        logger.info("Attempting to restart custom engine server...")
        
        # Stop the current server process
        if self.server_process and self.server_process.poll() is None:
            try:
                logger.info(f"Stopping custom engine process (PID: {self.server_process.pid})...")
                self.server_process.terminate()
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.server_process.wait(timeout=30)
                    logger.info("Custom engine stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Custom engine did not stop gracefully, force killing...")
                    self.server_process.kill()
                    self.server_process.wait()
                    logger.info("Custom engine force killed")
                    
            except Exception as e:
                logger.error(f"Error stopping custom engine: {str(e)}")
        
        # Start new server process
        logger.info("Starting new custom engine instance...")
        self.server_process = subprocess.Popen([sys.executable, self.startup_script_path])
        logger.info(f"Custom engine restarted with PID: {self.server_process.pid}")

