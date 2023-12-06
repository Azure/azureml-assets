# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copied from https://github.com/Azure/azureai-insiders/tree/main/previews/batch-inference-using-aoai
# and then modified.

from abc import abstractmethod

from .logging_utils import get_logger


class BaseLog:
    
    def __init__(self, internal_id, x_ms_client_request_id) -> None:
        self.internal_id = internal_id
        self.x_ms_client_request_id = x_ms_client_request_id

    @abstractmethod
    def log(self):
        pass

class ScoreStartLog(BaseLog):
    
    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 timestamp,
                 scoring_url,
                 ) -> None:
        super().__init__(internal_id, x_ms_client_request_id)
        self.timestamp = timestamp
        self.scoring_url = scoring_url

    def log(self):
        get_logger().info(f"Score start: url={self.scoring_url} internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")

class ScoreFailedLog(BaseLog):
    
    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 status_code,
                 reason,
                 response_headers,
                 response_payload) -> None:
        super().__init__(internal_id, x_ms_client_request_id)
        self.status_code = status_code
        self.reason = reason
        self.response_headers = response_headers
        self.response_payload = response_payload

    def log(self):
        get_logger().error(f"Score failed: status_code={self.status_code}, reason={self.reason} response_headers={self.response_headers} response_payload={self.response_payload}  internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreFailedWithExceptionLog(BaseLog):

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 exception_type,
                 exception,
                 unhandled_exc=False
                 ) -> None:
        super().__init__(internal_id, x_ms_client_request_id)
        self.exception_type = exception_type
        self.exception = exception
        self.unhandled_exc=unhandled_exc

    def log(self):
        get_logger().error(f"Score failed with exception: unhandled_exception={self.unhandled_exc} exception_type={self.exception_type}, exception={self.exception} internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreSucceedLog(BaseLog):
    
    def __init__(self, internal_id, x_ms_client_request_id, duration) -> None:
        super().__init__(internal_id, x_ms_client_request_id)
        self.duration = duration

    def log(self):
        get_logger().info(f"Score succeeded after {self.duration:.3f}s: internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")