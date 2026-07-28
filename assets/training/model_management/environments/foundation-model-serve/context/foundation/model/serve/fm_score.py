# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This module provides the FMScore class for running inference engines with given prompts and parameters."""

from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.replica_manager import ReplicaManager

logger = configure_logger(__name__)


class FMScore:
    """Class for running inference engines with given prompts and parameters."""

    def __init__(self):
        """Initialize the FMScore with the given configuration."""
        pass

    def init(self):
        """Initialize the engine and formatter."""
        try:
            self.replica_manager = self._init_replica_manager()
        except Exception as e:
            logger.error(f"Failed to create Engine with exception {repr(e)}")
            raise e

    def _init_replica_manager(self):
        replica_manager = ReplicaManager()
        replica_manager.initialize()
        return replica_manager

    # async def shutdown_async(self):
    #     """Terminate DS-MII Server."""
    #     await self.replica_manager.get_replica().shutdown_async()


if __name__ == "__main__":
    fms = FMScore()
    fms.init()
