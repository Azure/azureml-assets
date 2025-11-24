# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base engine module for inference backends.

This module defines the abstract base class for all inference engine implementations,
providing common functionality for server initialization and health checks.
"""

import time
import os
import socket
import requests
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

    def check_health_endpoint(self, host: str = CommonConstants.HOST,
                             port: int = CommonConstants.DEFAULT_PORT,
                             health_path: str = "/health",
                             timeout: float = 5.0) -> bool:
        """Check if the downstream engine health endpoint returns 200 OK.

        Args:
            host (str): The hostname or IP address to check.
            port (int): The port number to check.
            health_path (str): The health endpoint path (default: /health).
            timeout (float): Request timeout in seconds.

        Returns:
            bool: True if health endpoint returns 200 OK, False otherwise.
        """
        try:
            url = f"http://{host}:{port}{health_path}"
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"Health endpoint returned status {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.debug(f"Connection error when checking health endpoint at {url}")
            return False
        except requests.exceptions.Timeout:
            logger.warning(f"Health endpoint timeout at {url}")
            return False
        except Exception as e:
            logger.warning(f"Health check failed with error: {str(e)}")
            return False

    @log_execution_time
    def wait_until_server_healthy(self, host: str, port: int, health_check_timeout: float = 5.0):
        """Wait until the server health endpoint returns 200 OK.

        This method calls the downstream engine's /health endpoint to verify the server
        is healthy and ready to accept requests. It performs health checks every 30 seconds
        until the server is healthy or the maximum wait time (15 minutes) is exceeded.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number of the server.
            health_check_timeout (float): Timeout in seconds for each individual health check request (default: 5.0).

        Raises:
            Exception: If the server does not become healthy within 15 minutes.
        """
        start_time = time.time()
        health_check_interval = 30  # seconds between health checks
        max_wait_time = 15 * 60  # 15 minutes total
        
        should_log = os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid())
        
        if should_log:
            logger.info(
                f"Starting health checks for {host}:{port} "
                f"(checking every {health_check_interval}s, max wait {max_wait_time // 60} minutes)"
            )
        
        while time.time() - start_time < max_wait_time:
            # Check health endpoint
            is_healthy = self.check_health_endpoint(host, port, timeout=health_check_timeout)
            
            if is_healthy:
                elapsed = int(time.time() - start_time)
                if should_log:
                    logger.info(f"Server is healthy after {elapsed}s (health endpoint returned 200 OK).")
                return
            
            elapsed = int(time.time() - start_time)
            if should_log:
                logger.info(
                    f"Server not ready yet. Waiting {health_check_interval}s before next check... "
                    f"({elapsed}s/{max_wait_time}s elapsed)"
                )
            
            time.sleep(health_check_interval)
        
        raise Exception(
            f"Server health endpoint did not return 200 OK within {max_wait_time // 60} minutes."
        )
