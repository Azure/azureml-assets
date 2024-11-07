"""For context."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid
import time


class CorrelationContext:
    """For CorrelationContext."""

    def __init__(self):
        """For init."""
        pass

    @staticmethod
    def get_id() -> str:
        """For get id."""
        return ""

    def __str__(self):
        """For str."""
        return self.get_id()

    def __repr__(self):
        """For repr."""
        return self.__str__()


class BasicCorrelationContext(CorrelationContext):
    """For BasicCorrelationContext."""

    # pylint: disable=redefined-builtin
    def __init__(self, id: str = None, timestamp: int = None, headers=None):
        """For init."""
        super().__init__()
        self.id = id if id else str(uuid.uuid4())
        self.timestamp = timestamp if timestamp else int(time.time())
        self.headers = headers if headers else {}
    # pylint: enable=redefined-builtin

    def get_id(self) -> str:
        """For get id."""
        return self.id

    def get_timestamp(self) -> int:
        """For get timestamp."""
        return self.timestamp

    def get_headers(self) -> dict:
        """For get headers."""
        return self.headers


class WrapperContext(CorrelationContext):
    """For WrapperContext."""

    def __init__(self, correlation_context: CorrelationContext, success: bool, message: str):
        """For init."""
        super().__init__()
        self._context = correlation_context
        self._success = success
        self._message = message

    def get_id(self) -> str:
        """For get id."""
        return self._context.get_id()

    def get_timestamp(self) -> int:
        """For get timestamp."""
        return self._context.get_timestamp()

    def get_headers(self) -> dict:
        """For get headers."""
        return self._context.get_headers()

    def is_success(self) -> bool:
        """For is success."""
        return self._success

    def get_message(self) -> str:
        """For get message."""
        return self._message


def get_context() -> CorrelationContext:
    """For get context."""
    return BasicCorrelationContext()


# test purpose
def get_context_wrapper(context: CorrelationContext, success: bool, message: str) -> CorrelationContext:
    """For get context wrapper."""
    return WrapperContext(context, success, message)
