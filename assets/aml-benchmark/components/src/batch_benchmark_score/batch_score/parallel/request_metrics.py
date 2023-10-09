# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for request metrics."""

import pandas as pd


class RequestMetrics:
    """Class for request metrics."""

    COLUMN_TIMESTAMP = "timestamp"
    COLUMN_REQUEST_ID = "request_id"
    COLUMN_RESPONSE_CODE = "response_code"
    COLUMN_RESPONSE_PAYLOAD = "response_payload"
    COLUMN_MODEL_RESPONSE_CODE = "model_response_code"
    COLUMN_MODEL_RESPONSE_REASON = "model_response_reason"
    COLUMN_ADDITIONAL_WAIT_TIME = "additional_wait_time"
    COLUMN_REQUEST_TOTAL_WAIT_TIME = "request_total_wait_time"

    def __init__(self):
        """Init method."""
        self.__df = pd.DataFrame(columns=[
            RequestMetrics.COLUMN_TIMESTAMP,
            RequestMetrics.COLUMN_REQUEST_ID,
            RequestMetrics.COLUMN_RESPONSE_CODE,
            RequestMetrics.COLUMN_RESPONSE_PAYLOAD,
            RequestMetrics.COLUMN_MODEL_RESPONSE_CODE,
            RequestMetrics.COLUMN_MODEL_RESPONSE_REASON,
            RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME,
            RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME
        ])
        self.__df.set_index(RequestMetrics.COLUMN_TIMESTAMP, inplace=True)

    def add_result(
            self,
            request_id: str,
            response_code: int,
            response_payload: any,
            model_response_code: str,
            model_response_reason: str,
            additional_wait_time: int,
            request_total_wait_time: int
    ) -> None:
        """Add result."""
        self.__df.loc[pd.Timestamp.utcnow()] = [
            request_id, response_code, response_payload, model_response_code,
            model_response_reason, additional_wait_time, request_total_wait_time]

    def get_metrics(self, start_time: pd.Timestamp, end_time: pd.Timestamp = None) -> pd.DataFrame:
        """Get metrics."""
        if end_time is None:
            end_time = pd.Timestamp.utcnow()
        # NOTE: This only works on desc sorted data. self.__df is sorted in desc by nature.
        return self.__df.loc[start_time:end_time]
