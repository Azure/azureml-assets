# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for request modifier."""

from abc import ABC, abstractmethod


class RequestModifier(ABC):
    """Request modifier base class."""

    @abstractmethod
    def modify(self, request_obj: any) -> any:
        """Modify interface."""
        pass


class RequestModificationException(Exception):
    """Request modification exception class."""

    def __init__(
            self,
            message: str = "An exception was thrown while attempting to apply a RequestModifier."
    ) -> None:
        """Init class."""
        super().__init__(message)
