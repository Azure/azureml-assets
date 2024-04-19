# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import uuid
import time


class CorrelationContext:
    def __init__(self):
        pass

    @staticmethod
    def get_id() -> str:
        return ""

    def __str__(self):
        return self.get_id()

    def __repr__(self):
        return self.__str__()


class BasicCorrelationContext(CorrelationContext):
    # pylint: disable=redefined-builtin
    def __init__(self, id: str = None, timestamp: int = None, headers=None):
        super().__init__()
        self.id = id if id else str(uuid.uuid4())
        self.timestamp = timestamp if timestamp else int(time.time())
        self.headers = headers if headers else {}
    # pylint: enable=redefined-builtin

    def get_id(self) -> str:
        return self.id

    def get_timestamp(self) -> int:
        return self.timestamp

    def get_headers(self) -> dict:
        return self.headers


class WrapperContext(CorrelationContext):
    def __init__(self, correlation_context: CorrelationContext, success: bool, message: str):
        super().__init__()
        self._context = correlation_context
        self._success = success
        self._message = message

    def get_id(self) -> str:
        return self._context.get_id()

    def get_timestamp(self) -> int:
        return self._context.get_timestamp()

    def get_headers(self) -> dict:
        return self._context.get_headers()

    def is_success(self) -> bool:
        return self._success

    def get_message(self) -> str:
        return self._message


def get_context() -> CorrelationContext:
    return BasicCorrelationContext()


# test purpose
def get_context_wrapper(context: CorrelationContext, success: bool, message: str) -> CorrelationContext:
    return WrapperContext(context, success, message)
