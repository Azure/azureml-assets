# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for request metrics."""

import time

import pandas as pd

from src.batch_score.common.parallel.adjustment import RequestMetrics


class TestRequestMetrics:
    def test_add_result_and_get_metrics(self):
        """Test add result and get metrics."""
        request_metrics = RequestMetrics()
        start_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request1")

        metrics = request_metrics.get_metrics(
            start_time=start_time
        )

        assert len(metrics) == 1
        assert metrics.iloc[0][RequestMetrics.COLUMN_REQUEST_ID] == "request1"
        assert metrics.iloc[0][RequestMetrics.COLUMN_RESPONSE_CODE] == 200
        assert metrics.iloc[0][RequestMetrics.COLUMN_RESPONSE_PAYLOAD] == "response_payload"
        assert metrics.iloc[0][RequestMetrics.COLUMN_MODEL_RESPONSE_CODE] == "429"
        assert metrics.iloc[0][RequestMetrics.COLUMN_MODEL_RESPONSE_REASON] == "model_response_reason"
        assert metrics.iloc[0][RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME] == 10
        assert metrics.iloc[0][RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME] == 20

    def test_add_result_and_get_metrics_with_end_time(self):
        """Test add result and get metrics with end time."""
        request_metrics = RequestMetrics()
        start_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request1")
        self._add_request(request_metrics, "request2")

        end_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request3", delay=1)

        metrics = request_metrics.get_metrics(
            start_time=start_time,
            end_time=end_time,
        )

        # Only the first two metrics are returned.
        assert len(metrics) == 2

    def _add_request(self, request_metrics, request_id, delay=0):
        time.sleep(delay)
        request_metrics.add_result(
            request_id=request_id,
            response_code=200,
            response_payload="response_payload",
            model_response_code="429",
            model_response_reason="model_response_reason",
            additional_wait_time=10,
            request_total_wait_time=20,
        )

    def test_init_using_valid_dataframe(self):
        """Test init using valid dataframe."""
        request_metrics = RequestMetrics()
        start_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request1")

        # Initialize a new RequestMetrics object with the metrics from the first one.
        request_metrics_2 = RequestMetrics(metrics=request_metrics.get_metrics(start_time=start_time))

        metrics = request_metrics_2.get_metrics(
            start_time=start_time
        )

        assert len(metrics) == 1
        assert metrics.iloc[0][RequestMetrics.COLUMN_REQUEST_ID] == "request1"
        assert metrics.iloc[0][RequestMetrics.COLUMN_RESPONSE_CODE] == 200
        assert metrics.iloc[0][RequestMetrics.COLUMN_RESPONSE_PAYLOAD] == "response_payload"
        assert metrics.iloc[0][RequestMetrics.COLUMN_MODEL_RESPONSE_CODE] == "429"
        assert metrics.iloc[0][RequestMetrics.COLUMN_MODEL_RESPONSE_REASON] == "model_response_reason"
        assert metrics.iloc[0][RequestMetrics.COLUMN_ADDITIONAL_WAIT_TIME] == 10
        assert metrics.iloc[0][RequestMetrics.COLUMN_REQUEST_TOTAL_WAIT_TIME] == 20

    def test_init_using_dataframe_with_invalid_columns(self):
        """Test init using dataframe with invalid columns."""
        request_metrics = RequestMetrics()
        start_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request1")

        metrics = request_metrics.get_metrics(start_time=start_time)

        # Drop a column to make the dataframe invalid.
        metrics.drop(columns=[RequestMetrics.COLUMN_REQUEST_ID], inplace=True)

        # Initialize a new RequestMetrics object with the metrics from the first one.
        try:
            _ = RequestMetrics(metrics=metrics)
        except ValueError as e:
            assert 'Expected columns' in str(e)
        else:
            assert False  # If a ValueError exception wasn't raised, the test failed.

    def test_init_using_dataframe_with_invalid_index(self):
        """Test init using dataframe with invalid index."""
        request_metrics = RequestMetrics()
        start_time = pd.Timestamp.utcnow()

        self._add_request(request_metrics, "request1")

        metrics = request_metrics.get_metrics(start_time=start_time)

        # Change the index name to make the dataframe invalid.
        metrics['new column'] = 'some value'
        metrics.set_index('new column', inplace=True)

        # Initialize a new RequestMetrics object with the metrics from the first one.
        try:
            _ = RequestMetrics(metrics=metrics)
        except ValueError as e:
            assert 'Expected index name' in str(e)
        else:
            assert False  # If a ValueError exception wasn't raised, the test failed.
