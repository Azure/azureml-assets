# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copied from https://github.com/Azure/azureai-insiders/tree/main/previews/batch-inference-using-aoai
# and then modified.

"""Scoring logging helpers."""

from abc import abstractmethod

from .logging_utils import get_logger


class BaseLog:
    """Base log."""

    def __init__(self, internal_id, x_ms_client_request_id) -> None:
        """Init function."""
        self.internal_id = internal_id
        self.x_ms_client_request_id = x_ms_client_request_id

    @abstractmethod
    def log(self):
        """Log function."""
        pass


class ScoreStartLog(BaseLog):
    """Log a message for score start."""

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 timestamp,
                 scoring_url,
                 ) -> None:
        """Init function."""
        super().__init__(internal_id, x_ms_client_request_id)
        self.timestamp = timestamp
        self.scoring_url = scoring_url

    def log(self):
        """Log function."""
        msg = "Score start: url={} internal_id={} x-ms-client-request-id=[{}]"
        get_logger().info(msg.format(self.scoring_url,
                                     self.internal_id,
                                     self.x_ms_client_request_id))


class ScoreFailedLog(BaseLog):
    """Log a message for score failed."""

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 status_code,
                 reason,
                 response_headers,
                 response_payload) -> None:
        """Init function."""
        super().__init__(internal_id, x_ms_client_request_id)
        self.status_code = status_code
        self.reason = reason
        self.response_headers = response_headers
        self.response_payload = response_payload

    def log(self):
        """Log function."""
        get_logger().error(f"Score failed: status_code={self.status_code}, reason={self.reason} "
                           + f"response_headers={self.response_headers} response_payload={self.response_payload} "
                           + f"internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreFailedWithExceptionLog(BaseLog):
    """Log a message for score failed with an exception."""

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 exception_type,
                 exception,
                 unhandled_exc=False
                 ) -> None:
        """Init function."""
        super().__init__(internal_id, x_ms_client_request_id)
        self.exception_type = exception_type
        self.exception = exception
        self.unhandled_exc = unhandled_exc

    def log(self):
        """Log function."""
        get_logger().error(f"Score failed with exception: unhandled_exception={self.unhandled_exc} "
                           + f"exception_type={self.exception_type}, exception={self.exception} "
                           + f"internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreSucceedLog(BaseLog):
    """Log a message for score succeed."""

    def __init__(self, internal_id, x_ms_client_request_id, duration) -> None:
        """Init function."""
        super().__init__(internal_id, x_ms_client_request_id)
        self.duration = duration

    def log(self):
        """Log function."""
        msg = "Score succeeded after {:.3f}s: internal_id={} x-ms-client-request-id=[{}]"
        get_logger().info(msg.format(self.duration, self.internal_id, self.x_ms_client_request_id))
