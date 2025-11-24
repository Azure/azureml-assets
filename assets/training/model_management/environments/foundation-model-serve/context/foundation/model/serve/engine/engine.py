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

    def __init__(self):
        """Initialize the BaseEngine with configurable retry settings."""
        # Number of consecutive port check failures before attempting restart
        self.checks_before_restart = int(os.environ.get("ENGINE_CHECKS_BEFORE_RESTART", "5"))
        # Maximum number of restart attempts
        self.max_restart_attempts = int(os.environ.get("ENGINE_MAX_RESTART_ATTEMPTS", "3"))
        # Interval between port checks in seconds
        self.check_interval = int(os.environ.get("ENGINE_CHECK_INTERVAL", "30"))
        # Wait time after restart before resuming checks
        self.restart_wait_time = int(os.environ.get("ENGINE_RESTART_WAIT_TIME", "60"))

    @abstractmethod
    def init_server(self):
        """Initialize client[s] for the engine to receive requests on.

        This method should be implemented by subclasses to start the engine server.
        """
        pass

    @abstractmethod
    def restart_server(self):
        """Restart the engine server.

        This method should be implemented by subclasses to restart the server
        when health checks fail repeatedly. If restart is not supported,
        raise NotImplementedError.

        Raises:
            NotImplementedError: If restart functionality is not implemented.
        """
        raise NotImplementedError("Server restart not implemented for this engine type.")

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
        """Wait until the server is healthy and accepting connections with automatic restart capability.

        This method checks if the port is open. After consecutive failures, it attempts to restart
        the server. The number of checks before restart and maximum restart attempts are configurable.

        Args:
            host (str): The hostname or IP address of the server.
            port (int): The port number of the server.
            timeout (float): Connection timeout in seconds for each check.

        Raises:
            Exception: If the server does not become healthy within 15 minutes or after all restart attempts.

        Environment Variables:
            ENGINE_CHECKS_BEFORE_RESTART (int): Number of consecutive failures before restart (default: 5)
            ENGINE_MAX_RESTART_ATTEMPTS (int): Maximum restart attempts (default: 3)
            ENGINE_CHECK_INTERVAL (int): Seconds between checks (default: 30)
            ENGINE_RESTART_WAIT_TIME (int): Seconds to wait after restart (default: 60)
        """
        start_time = time.time()
        max_wait_time = 1 * 60  # 15 minutes total
        
        consecutive_failures = 0
        restart_attempt = 0
        
        should_log = os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid())
        
        while time.time() - start_time < max_wait_time:
            is_healthy = self.is_port_open(host, port, timeout)
            
            if is_healthy:
                # Server is healthy - reset failure counter and return
                if should_log:
                    if consecutive_failures > 0:
                        logger.info(f"Server recovered after {consecutive_failures} failed checks.")
                    logger.info("Server is healthy.")
                return
            
            # Port check failed
            consecutive_failures += 1
            elapsed = int(time.time() - start_time)
            
            if should_log:
                logger.warning(
                    f"Port {port} check failed (attempt {consecutive_failures}/{self.checks_before_restart}). "
                    f"Elapsed time: {elapsed}s"
                )
            
            # Check if we should attempt a restart
            if consecutive_failures >= self.checks_before_restart:
                if restart_attempt < self.max_restart_attempts:
                    restart_attempt += 1
                    
                    if should_log:
                        logger.warning(
                            f"Server unhealthy after {consecutive_failures} consecutive checks. "
                            f"Attempting restart {restart_attempt}/{self.max_restart_attempts}..."
                        )
                    
                    try:
                        # Attempt to restart the server
                        self.restart_server()
                        
                        if should_log:
                            logger.info(
                                f"Restart command executed. Waiting {self.restart_wait_time}s "
                                f"for server to initialize..."
                            )
                        
                        # Wait for server to initialize after restart
                        time.sleep(self.restart_wait_time)
                        
                        # Reset consecutive failures after restart attempt
                        consecutive_failures = 0
                        
                        # Continue checking
                        continue
                        
                    except NotImplementedError:
                        if should_log:
                            logger.warning(
                                "Server restart not implemented for this engine type. "
                                "Continuing with health checks..."
                            )
                        # Reset counter to avoid continuous restart attempts
                        consecutive_failures = 0
                        
                    except Exception as e:
                        if should_log:
                            logger.error(f"Failed to restart server: {str(e)}")
                        # Reset counter and continue checking
                        consecutive_failures = 0
                else:
                    # Exceeded max restart attempts
                    if should_log:
                        logger.error(
                            f"Server did not recover after {self.max_restart_attempts} restart attempts. "
                            f"Continuing health checks until timeout..."
                        )
                    # Reset counter to avoid continuous restart attempts
                    consecutive_failures = 0
            
            # Wait before next check
            if should_log:
                logger.info(f"Waiting {self.check_interval}s before next health check...")
            time.sleep(self.check_interval)
        
        # Timeout reached
        raise Exception(
            f"Server did not become healthy within {max_wait_time // 60} minutes. "
            f"Restart attempts: {restart_attempt}/{self.max_restart_attempts}"
        )
