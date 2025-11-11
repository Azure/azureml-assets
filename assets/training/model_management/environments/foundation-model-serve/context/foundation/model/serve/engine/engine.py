# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base engine module for inference backends.

This module defines the abstract base class for all inference engine implementations,
providing common functionality for server initialization and health checks.
"""

import time
import os
import socket
from abc import ABC, abstractmethod
from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.utils import log_execution_time
from foundation.model.serve.constants import CommonConstants

logger = configure_logger(__name__)


class BaseEngine(ABC):
    """Abstract base class for inference engine backends.
    
    This class defines the interface that all inference engines must implement,
    including server initialization and health monitoring capabilities.
    """

    @abstractmethod
    def init_server(self):
        """Initialize client[s] for the engine to receive requests on.
        
        This method should be implemented by subclasses to start the engine server.
        """
        pass

    def is_port_open(self, host: str = CommonConstants.HOST,
                     port: int = CommonConstants.DEFAULT_PORT, timeout: float = 1.0) -> bool:
        """Check if a port is open on the given host.
        
        Args:
            host (str): The hostname or IP address to check.
            port (int): The port number to check.
            timeout (float): Connection timeout in seconds.
            
        Returns:
            bool: True if the port is open and accepting connections, False otherwise.
        """
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    @log_execution_time
    def wait_until_server_healthy(self, host: str, port: int, timeout: float = 1.0):
        """Wait until the server is healthy and accepting connections.
        
        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number of the server.
            timeout (float): Connection timeout in seconds for each check.
            
        Raises:
            Exception: If the server does not become healthy within 15 minutes.
        """
        start_time = time.time()
        while time.time() - start_time < 15 * 60:
            is_healthy = self.is_port_open(host, port, timeout)
            if is_healthy:
                if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                    logger.info("Server is healthy.")
                return
            if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                logger.info("Waiting for server to start...")
            time.sleep(30)
        raise Exception("Server did not become healthy within 15 minutes.")
