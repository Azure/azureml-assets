# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copied from https://github.com/Azure/azureai-insiders/tree/main/previews/batch-inference-using-aoai
# and then modified.

from abc import abstractmethod

from .logging_utils import get_logger


class BaseLog:

    def __init__(self, internal_id, x_ms_client_request_id, scoring_url) -> None:
        self.internal_id = internal_id
        self.x_ms_client_request_id = x_ms_client_request_id
        self.scoring_url = scoring_url

    @abstractmethod
    def log(self):
        pass


class ScoreStartLog(BaseLog):

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 scoring_url,
                 ) -> None:
        super().__init__(internal_id, x_ms_client_request_id, scoring_url)

    def log(self):
        get_logger().info(f"Score start: url={self.scoring_url} internal_id={self.internal_id} "
                          f"x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreFailedLog(BaseLog):

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 scoring_url,
                 status_code,
                 reason,
                 response_headers,
                 response_payload) -> None:
        super().__init__(internal_id, x_ms_client_request_id, scoring_url)
        self.status_code = status_code
        self.reason = reason
        self.response_headers = response_headers
        self.response_payload = response_payload

    def log(self):
        get_logger().error(f"Score failed: status_code={self.status_code}, reason={self.reason} "
                           f"response_headers={self.response_headers} response_payload={self.response_payload} "
                           f"url={self.scoring_url} internal_id={self.internal_id} "
                           f"x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreFailedWithExceptionLog(BaseLog):

    def __init__(self,
                 internal_id,
                 x_ms_client_request_id,
                 scoring_url,
                 exception_type,
                 exception,
                 unhandled_exc=False
                 ) -> None:
        super().__init__(internal_id, x_ms_client_request_id, scoring_url)
        self.exception_type = exception_type
        self.exception = exception
        self.unhandled_exc = unhandled_exc

    def log(self):
        get_logger().error(f"Score failed with exception: unhandled_exception={self.unhandled_exc} "
                           f"exception_type={self.exception_type}, exception={self.exception} "
                           f"url={self.scoring_url} internal_id={self.internal_id} "
                           f"x-ms-client-request-id=[{self.x_ms_client_request_id}]")


class ScoreSucceedLog(BaseLog):

    def __init__(self, internal_id, x_ms_client_request_id, scoring_url, duration) -> None:
        super().__init__(internal_id, x_ms_client_request_id, scoring_url)
        self.duration = duration

    def log(self):
        get_logger().info(f"Score succeeded after {self.duration:.3f}s: url={self.scoring_url} "
                          f"internal_id={self.internal_id} x-ms-client-request-id=[{self.x_ms_client_request_id}]")
