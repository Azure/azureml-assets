# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Request modifier base class."""

from abc import ABC, abstractmethod


class RequestModifier(ABC):
    """Request modifier."""

    @abstractmethod
    def modify(self, request_obj: any) -> any:
        """Modify the request object."""
        pass


class RequestModificationException(Exception):
    """Request modifier exception."""

    def __init__(self, message: str = "An exception was thrown while attempting to apply a RequestModifier.") -> None:
        """Initialize RequestModificationException."""
        super().__init__(message)
