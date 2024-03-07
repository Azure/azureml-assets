# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for Gen AI preprocessor."""

from pyspark.sql.types import (
    StructField, StringType, TimestampType, StructType
)
import pytest
import os
import sys
from datetime import datetime
from src.model_data_collector_preprocessor.trace_aggregator import (
    aggregate_spans_into_traces,
)
from src.shared_utilities.io_utils import init_spark


@pytest.fixture(scope="module")
def genai_preprocessor_test_setup():
    # TODO: move this test utility to shared conftest later.
    # Check gsq tests and mdc tests for possible duplications.
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
    os.environ["PYTHONPATH"] = old_python_path  # change python path back to original


@pytest.mark.unit
class TestGenAISparkPreprocessor:
    """Test class for Gen AI Preprocessor."""

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

    _span_log_data_extra = [
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "in", "[]", "name",  "out", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "in", "[]", "name",  "out", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ["resource", "c", "{}", datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "in", "[]", "name",  "out", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    _root_span_str_extra = '{"attributes": "resource", "context": "c", "c_resources": "{}", "end_time": "2024-02-' + \
        '05T00:08:00", "events": "[]", "framework": "FLOW", "input": "in", "links": "[]", "name": "name", "output"' + \
        ': "out", "parent_id": null, "span_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "sta' + \
        'tus": "OK", "trace_id": "01", "children": [{"attributes": "resource", "context": "c", "c_resources": "{}"' + \
        ', "end_time": "2024-02-05T00:05:00", "events": "[]", "framework": "RAG", "input": "in", "links": "[]", "n' + \
        'ame": "name", "output": "out", "parent_id": "1", "span_id": "2", "span_type": "llm", "start_time": "2024' + \
        '-02-05T00:02:00", "status": "OK", "trace_id": "01", "children": [{"attributes": "resource", "context": "' + \
        'c", "c_resources": "{}", "end_time": "2024-02-05T00:04:00", "events": "[]", "framework": "INTERNAL", "in' + \
        'put": "in", "links": "[]", "name": "name", "output": "out", "parent_id": "2", "span_id": "3", "span_type' + \
        '": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "children": []}]}, {"at' + \
        'tributes": "resource", "context": "c", "c_resources": "{}", "end_time": "2024-02-05T00:07:00", "events":' + \
        ' "[]", "framework": "LLM", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": "' + \
        '1", "span_id": "4", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id":' + \
        ' "01", "children": []}]}'

    _root_span_str = '{"attributes": "{}", "end_time": "2024-02-05T00:08:00", "events": "[]", "framework": "FL' + \
        'OW", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": null, "span_id": "1",' + \
        ' "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01", "children"' + \
        ': [{"attributes": "{}", "end_time": "2024-02-05T00:05:00", "event' + \
        's": "[]", "framework": "RAG", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_i' + \
        'd": "1", "span_id": "2", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "tra' + \
        'ce_id": "01", "children": [{"attributes": "{}", "end_time": "2024-02-05T00:04:00", "events": "[]", "fr' + \
        'amework": "INTERNAL", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": "2",' + \
        ' "span_id": "3", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": ' + \
        '"01", "children": []}]}, {"attributes": "{}", "end_time": "2024-02-05T00:07:00", "events": "[]", "fram' + \
        'ework": "LLM", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": "1", "span_' + \
        'id": "4", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01", "' + \
        'children": []}]}'

    _trace_log_data = [
            ["01", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str],
    ]

    _trace_log_schema = StructType(
        [
            StructField("trace_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
        ]
    )

    _trace_log_data_extra = [
            ["01", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_extra],
    ]

    # 6-7 is desired hour
    # trace 00 is before the desired time.
    # trace 01 has child in hour before
    # trace 02
    _span_log_data_lookback = [
        ["{}", datetime(2024, 2, 5, 5, 10), "[]", "FLOW", "in", "[]", "name",  "out", None] +
        ["1", "llm", datetime(2024, 2, 5, 5, 15, 0), "OK", "00"],
        ["{}", datetime(2024, 2, 5, 5, 59), "[]", "RAG", "in", "[]", "name",  "out", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 5, 58, 0), "OK", "00"],
        ["{}", datetime(2024, 2, 5, 5, 59, 0), "[]", "INTERNAL", "in", "[]", "name",  "out", "4"] +
        ["3", "llm", datetime(2024, 2, 5, 5, 58, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 5, 58, 0), "[]", "LLM", "in", "[]", "name",  "out", "5"] +
        ["4", "llm", datetime(2024, 2, 5, 6, 5, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 6, 59, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
        ["5", "llm", datetime(2024, 2, 5, 6, 8, 0), "OK", "01"],
        ["{}", datetime(2024, 2, 5, 6, 58, 0), "[]", "RAG", "in", "[]", "name",  "out", "6"] +
        ["6", "llm", datetime(2024, 2, 5, 6, 50, 0), "OK", "02"],
        ["{}", datetime(2024, 2, 5, 7, 4, 0), "[]", "INTERNAL", "in", "[]", "name",  "out", "8"] +
        ["7", "llm", datetime(2024, 2, 5, 6, 59, 0), "OK", "03"],
        ["{}", datetime(2024, 2, 5, 7, 7, 0), "[]", "LLM", "in", "[]", "name",  "out", "9"] +
        ["8", "llm", datetime(2024, 2, 5, 7, 6, 0), "OK", "03"]
    ]

    _root_span_str_lookback = '{"attributes": "{}", "end_time": "2024-02-05T06:59:00", "events": "[]", "framewor' + \
        'k": "FLOW", "input": "in", "links": "[]", "name": "name", "output": "out", "parent_id": null, "span_id"' + \
        ': "5", "span_type": "llm", "start_time": "2024-02-05T06:08:00", "status": "OK", "trace_id": "01", "chil' + \
        'dren": [{"attributes": "{}", "end_time": "2024-02-05T05:58:00", "events": "[]", "framework": "LLM", "in' + \
        'put": "in", "links": "[]", "name": "name", "output": "out", "parent_id": "5", "span_id": "4", "span_typ' + \
        'e": "llm", "start_time": "2024-02-05T06:05:00", "status": "OK", "trace_id": "01", "children": [{"attrib' + \
        'utes": "{}", "end_time": "2024-02-05T05:59:00", "events": "[]", "framework": "INTERNAL", "input": "in",' + \
        ' "links": "[]", "name": "name", "output": "out", "parent_id": "4", "span_id": "3", "span_type": "llm", ' + \
        '"start_time": "2024-02-05T05:58:00", "status": "OK", "trace_id": "01", "children": []}]}]}'

    _trace_log_data_lookback = [
            ["01", None, None, datetime(2024, 2, 5, 6, 8, 0)] +
            [datetime(2024, 2, 5, 6, 59, 0), "in", "out", _root_span_str_lookback],
    ]

    def test_trace_aggregator_empty_root_span(self, code_zip_test_setup, genai_preprocessor_test_setup):
        """Test scenarios where we have a faulty root span when generating tree."""
        spark = init_spark()
        start_time = datetime(2024, 2, 5, 0, 1, 0)
        end_time = datetime(2024, 2, 5, 0, 8, 0)

        span_logs_no_root_with_data = [
            ["{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", None] +
            ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
            ["{}", datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "in", "[]", "name",  "out", "1"] +
            ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "02"],
        ]
        span_logs_no_root_with_data_df = spark.createDataFrame(
            span_logs_no_root_with_data,
            self._preprocessed_log_schema)

        trace_df = aggregate_spans_into_traces(span_logs_no_root_with_data_df, True, start_time, end_time)
        rows = trace_df.collect()
        assert trace_df.count() == 1
        assert rows[0]['trace_id'] == "01"

        span_logs_no_root = [
            ["{}", datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "in", "[]", "name",  "out", "1"] +
            ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ]
        spans_no_root_df = spark.createDataFrame(span_logs_no_root, self._preprocessed_log_schema)
        no_root_traces = aggregate_spans_into_traces(spans_no_root_df, True, start_time, end_time)
        assert no_root_traces.isEmpty()

    @pytest.mark.parametrize(
        "span_input_logs, span_input_schema, expected_trace_logs, " +
        "expected_trace_schema, require_trace_data, data_window_start, data_window_end",
        [
            ([], _preprocessed_log_schema, [], _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data, _preprocessed_log_schema, _trace_log_data, _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data_extra, _preprocessed_log_schema_extra, _trace_log_data_extra, _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data, _preprocessed_log_schema, [], _trace_log_schema, False,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            # Look back with extra span logs
            (_span_log_data_lookback, _preprocessed_log_schema, _trace_log_data_lookback, _trace_log_schema, True,
             datetime(2024, 2, 5, 6), datetime(2024, 2, 5, 7))
        ]
    )
    def test_trace_aggregator(
            self, code_zip_test_setup, genai_preprocessor_test_setup,
            span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema,
            require_trace_data, data_window_start, data_window_end):
        """Test scenario where spans has real data."""
        spark = init_spark()

        # infer schema only when we have data.
        processed_spans_df = spark.createDataFrame(span_input_logs, span_input_schema)
        expected_traces_df = spark.createDataFrame(expected_trace_logs, expected_trace_schema)

        print("processed logs:")
        processed_spans_df.show()
        processed_spans_df.printSchema()

        print("expected trace logs:")
        expected_traces_df.show()
        expected_traces_df.printSchema()

        actual_trace_df = aggregate_spans_into_traces(
            processed_spans_df, require_trace_data, data_window_start, data_window_end)

        print("actual trace logs:")
        actual_trace_df.show()
        actual_trace_df.printSchema()

        assert_spark_dataframe_equal(actual_trace_df, expected_traces_df)


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    print(f'df1: {df1.collect()}')
    print(f'df2: {df2.collect()}')
    assert df1.collect() == df2.collect()
