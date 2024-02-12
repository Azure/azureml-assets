# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the trace aggregator."""


from src.model_data_collector_preprocessor.trace_aggregator import (
    process_spans_into_aggregated_traces,
    _get_aggregated_trace_log_spark_df_schema
)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pytest


@pytest.mark.unit
class TestTraceAggregator:
    """Test class for Trace Aggregator."""

    def test_trace_aggregator(self):
        processed_span_logs = create_pyspark_dataframe([], _get_aggregated_trace_log_spark_df_schema().fieldNames())
        trace_logs = process_spans_into_aggregated_traces(processed_span_logs)
        assert trace_logs.isEmpty()
