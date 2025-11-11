# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import os
import socket
from abc import ABC, abstractmethod
from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.utils import log_execution_time
from foundation.model.serve.constants import CommonConstants

logger = configure_logger(__name__)


class BaseEngine(ABC):
    """Base class for inference engines backends."""

    @abstractmethod
    def init_server(self):
        """Initialize client[s] for the engine to receive requests on."""
        pass

    def is_port_open(self, host: str = CommonConstants.HOST, port: int = CommonConstants.DEFAULT_PORT, timeout: float = 1.0) -> bool:
        """Check if a port is open on the given host."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    @log_execution_time
    def wait_until_server_healthy(self, host: str, port: int, timeout: float = 1.0):
        """Wait until the server is healthy."""
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
