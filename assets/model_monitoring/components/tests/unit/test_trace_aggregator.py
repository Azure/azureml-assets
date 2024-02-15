# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the trace aggregator."""

import os
import pytest
import sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from src.model_data_collector_preprocessor.trace_aggregator import (
    process_spans_into_aggregated_traces,
)


@pytest.fixture(scope="module")
def genai_preprocessor_test_setup():
    """Change working directory to root of the assets/model_monitoring_components."""
    original_work_dir = os.getcwd()
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    os.chdir(momo_work_dir)  # change working directory to root of the assets/model_monitoring_components
    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    print("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", "NA"))
    module_path = f"{os.getcwd()}/src"
    old_python_path = os.environ.get("PYTHONPATH", None)
    old_python_path = f"{old_python_path};" if old_python_path else ""
    os.environ["PYTHONPATH"] = f"{old_python_path}{module_path}"
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NA"))
    yield
    os.chdir(original_work_dir)  # change working directory back to original


@pytest.mark.unit
class TestTraceAggregator:
    """Test class for Trace Aggregator."""

    def init_spark(self) -> SparkSession:
        """Create spark session."""
        return SparkSession.builder.appName("test").getOrCreate()

    _preprocessed_log_schema = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('input', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('output', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
    ])

    _span_log_data = [
        ["{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "in", "[]", "name",  "out", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "in", "[]", "name",  "out", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "in", "[]", "name",  "out", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    # pass data that might have extra information like context, resource, etc
    _preprocessed_log_schema_extra = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('context', StringType(), True),
        StructField('c_resources', StringType(), True),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('input', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('output', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
    ])

    _extra_info_span_log_data = [
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "in", "[]", "name",  "out", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "in", "[]", "name",  "out", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "in", "[]", "name",  "out", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    _root_span_str_extra = '{"attributes": "resource", "context": "c", "c_resources": "{}", "end_time": "2024-02-05T00:08:00", "events": "[]", "framework": "FLOW", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": null, "span_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01", "children": ["{\\"attributes\\": \\"resource\\", \\"context\\": \\"c\\", \\"c_resources\\": \\"{}\\", \\"end_time\\": \\"2024-02-05T00:05:00\\", \\"events\\": \\"[]\\", \\"framework\\": \\"RAG\\", \\"input\\": \\"in\\", \\"links\\": \\"[]\\", \\"name\\": \\"name\\", \\"output\\": \\"out\\", \\"parent_id\\": \\"1\\", \\"span_id\\": \\"2\\", \\"span_type\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:02:00\\", \\"status\\": \\"OK\\", \\"trace_id\\": \\"01\\", \\"children\\": [\\"{\\\\\\"attributes\\\\\\": \\\\\\"resource\\\\\\", \\\\\\"context\\\\\\": \\\\\\"c\\\\\\", \\\\\\"c_resources\\\\\\": \\\\\\"{}\\\\\\", \\\\\\"end_time\\\\\\": \\\\\\"2024-02-05T00:04:00\\\\\\", \\\\\\"events\\\\\\": \\\\\\"[]\\\\\\", \\\\\\"framework\\\\\\": \\\\\\"INTERNAL\\\\\\", \\\\\\"input\\\\\\": \\\\\\"in\\\\\\", \\\\\\"links\\\\\\": \\\\\\"[]\\\\\\", \\\\\\"name\\\\\\": \\\\\\"name\\\\\\", \\\\\\"output\\\\\\": \\\\\\"out\\\\\\", \\\\\\"parent_id\\\\\\": \\\\\\"2\\\\\\", \\\\\\"span_id\\\\\\": \\\\\\"3\\\\\\", \\\\\\"span_type\\\\\\": \\\\\\"llm\\\\\\", \\\\\\"start_time\\\\\\": \\\\\\"2024-02-05T00:03:00\\\\\\", \\\\\\"status\\\\\\": \\\\\\"OK\\\\\\", \\\\\\"trace_id\\\\\\": \\\\\\"01\\\\\\", \\\\\\"children\\\\\\": []}\\"]}", "{\\"attributes\\": \\"resource\\", \\"context\\": \\"c\\", \\"c_resources\\": \\"{}\\", \\"end_time\\": \\"2024-02-05T00:07:00\\", \\"events\\": \\"[]\\", \\"framework\\": \\"LLM\\", \\"input\\": \\"in\\", \\"links\\": \\"[]\\", \\"name\\": \\"name\\", \\"output\\": \\"out\\", \\"parent_id\\": \\"1\\", \\"span_id\\": \\"4\\", \\"span_type\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:06:00\\", \\"status\\": \\"OK\\", \\"trace_id\\": \\"01\\", \\"children\\": []}"]}'

    _root_span_str = '{"attributes": "{}", "end_time": "2024-02-05T00:08:00", "events": "[]", "framework": "FLOW",' + \
        ' "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": null, "span_id":' + \
        ' "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "0' + \
        '1", "children": ["{\\"attributes\\": \\"{}\\", \\"end_time\\": \\"2024-02-05T00:05:00\\", \\"e' + \
        'vents\\": \\"[]\\", \\"framework\\": \\"RAG\\", \\"input\\": \\"in\\", \\"links\\": \\"[]\\", \\"' + \
        'name\\": \\"name\\", \\"output\\": \\"out\\", \\"parent_id\\": \\"1\\", \\"span_id\\": \\"2\\"' + \
        ', \\"span_type\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:02:00\\", \\"status\\": \\"OK\\"' + \
        ', \\"trace_id\\": \\"01\\", \\"children\\": [\\"{\\\\\\"attributes\\\\\\": \\\\\\"{}\\\\\\"' + \
        ', \\\\\\"end_time\\\\\\": \\\\\\"2024-02-05T00:04:00\\\\\\", \\\\\\"events\\\\\\": \\\\\\"[]\\\\\\"' + \
        ', \\\\\\"framework\\\\\\": \\\\\\"INTERNAL\\\\\\", \\\\\\"input\\\\\\": \\\\\\"in\\\\\\", \\\\\\"li' + \
        'nks\\\\\\": \\\\\\"[]\\\\\\", \\\\\\"name\\\\\\": \\\\\\"name\\\\\\", \\\\\\"output\\\\\\": \\\\\\"' + \
        'out\\\\\\", \\\\\\"parent_id\\\\\\": \\\\\\"2\\\\\\", \\\\\\"span_id\\\\\\": \\\\\\"3\\\\\\", \\\\\\"' + \
        'span_type\\\\\\": \\\\\\"llm\\\\\\", \\\\\\"start_time\\\\\\": \\\\\\"2024-02-05T00:03:00\\\\\\",' + \
        ' \\\\\\"status\\\\\\": \\\\\\"OK\\\\\\", \\\\\\"trace_id\\\\\\": \\\\\\"01\\\\\\", \\\\\\"childre' + \
        'n\\\\\\": []}\\"]}", "{\\"attributes\\": \\"{}\\", \\"end_time\\": \\"2024-02-05T00:07:00\\", \\"even' + \
        'ts\\": \\"[]\\", \\"framework\\": \\"LLM\\", \\"input\\": \\"in\\", \\"links\\": \\"[]\\", \\"name\\"' + \
        ': \\"name\\", \\"output\\": \\"out\\", \\"parent_id\\": \\"1\\", \\"span_id\\": \\"4\\", \\"span_typ' + \
        'e\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:06:00\\", \\"status\\": \\"OK\\", \\"trace_id\\":' + \
        ' \\"01\\", \\"children\\": []}"]}'

    _trace_log_data = [
            ["{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
            ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01", _root_span_str],
    ]

    _trace_log_schema = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('input', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('output', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
        StructField('root_span', StringType(), False),
    ])

    _trace_log_data_extra = [
            ["resource", "c", "{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
            ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01", _root_span_str_extra],
    ]

    _trace_log_schema_extra = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('context', StringType(), True),
        StructField('c_resources', StringType(), True),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('input', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('output', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
        StructField('root_span', StringType(), False),
    ])

    @pytest.mark.parametrize(
            "span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema",
            [
                ([], _preprocessed_log_schema, [], _trace_log_schema),
                (_span_log_data, _preprocessed_log_schema, _trace_log_data, _trace_log_schema),
                (_extra_info_span_log_data, _preprocessed_log_schema_extra, _trace_log_data_extra, _trace_log_schema_extra),
            ]
    )
    def test_trace_aggregator(
            self, genai_preprocessor_test_setup,
            span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema):
        """Test scenario where spans has real data."""
        spark = self.init_spark()
        # infer schema only when we have data.
        processed_spans_df = spark.createDataFrame(span_input_logs, span_input_schema)
        expected_traces_df = spark.createDataFrame(expected_trace_logs, expected_trace_schema)

        print("processed logs:")
        processed_spans_df.show()
        processed_spans_df.printSchema()

        print("expected trace logs:")
        expected_traces_df.show()
        expected_traces_df.printSchema()

        actual_trace_df = process_spans_into_aggregated_traces(processed_spans_df)

        print("actual trace logs:")
        actual_trace_df.show()
        actual_trace_df.printSchema()

        assert_spark_dataframe_equal(actual_trace_df, expected_traces_df)


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    print(f"df1: {df1.collect()}")
    print(f"df2: {df2.collect()}")
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    assert df1.collect() == df2.collect()
