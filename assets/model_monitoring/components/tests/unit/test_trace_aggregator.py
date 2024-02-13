# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the trace aggregator."""

from pyspark.sql import SparkSession
from src.model_data_collector_preprocessor.trace_aggregator import (
    process_spans_into_aggregated_traces,
    _get_aggregated_trace_log_spark_df_schema
)
import pytest


@pytest.mark.unit
class TestTraceAggregator:
    """Test class for Trace Aggregator."""

    def init_spark(self)-> SparkSession:
        return SparkSession.builder.appName("test").getOrCreate()

    def test_trace_aggregator_empty(self):
        """Test scenario where data is empty."""
        spark = self.init_spark()
        processed_span_logs = spark.createDataFrame([], _get_aggregated_trace_log_spark_df_schema())
        trace_logs = process_spans_into_aggregated_traces(processed_span_logs)
        assert trace_logs.isEmpty()
