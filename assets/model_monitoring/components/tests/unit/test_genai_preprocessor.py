# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for Gen AI preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField, StringType, TimestampType, StructType
)
import pytest
import os
import sys
import string
import random
import time
import zipfile
from datetime import datetime
from model_data_collector_preprocessor.genai_run import (
    _genai_uri_folder_to_preprocessed_spark_df,
)
from model_data_collector_preprocessor.trace_aggregator import (
    process_spans_into_aggregated_traces,
)
from model_data_collector_preprocessor.store_url import StoreUrl
from spark_mltable import SPARK_ZIP_PATH


@pytest.fixture(scope="session")
def genai_zip_test_setup():
    # TODO: move this zip utility to shared conftest later.
    # Check gsq tests and mdc tests for possible duplications.
    """Zip files in module_path to src.zip."""
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    module_path = os.path.join(momo_work_dir, "src")
    # zip files in module_path to src.zip
    s = string.ascii_lowercase + string.digits
    rand_str = '_' + ''.join(random.sample(s, 5))
    time_str = time.strftime("%Y%m%d-%H%M%S") + rand_str
    zip_path = os.path.join(module_path, f"src_{time_str}.zip")

    zf = zipfile.ZipFile(zip_path, "w")
    for dirname, subdirs, files in os.walk(module_path):
        for filename in files:
            abs_filepath = os.path.join(dirname, filename)
            rel_filepath = os.path.relpath(abs_filepath, start=module_path)
            print("zipping file:", rel_filepath)
            zf.write(abs_filepath, arcname=rel_filepath)
    zf.close()
    # add files to zip folder
    os.environ[SPARK_ZIP_PATH] = zip_path
    print("zip path set in genai_preprocessor_test_setup: ", zip_path)

    yield
    # remove zip file
    os.remove(zip_path)
    # remove zip path from environment
    os.environ.pop(SPARK_ZIP_PATH, None)


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

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        return SparkSession.builder.appName("test").getOrCreate()

    _preprocessed_schema = StructType([
        StructField('trace_id', StringType(), True),
        StructField('context', StringType(), True),
        StructField('end_time', TimestampType(), True),
        StructField('events', StringType(), True),
        StructField('links', StringType(), True),
        StructField('name', StringType(), True),
        StructField('parent_id', StringType(), True),
        StructField('start_time', TimestampType(), True),
        StructField('status', StringType(), True),
        StructField('span_id', StringType(), True),
        StructField('span_type', StringType(), True),
        StructField('framework', StringType(), True),
        StructField('input', StringType(), True),
        StructField('output', StringType(), True),
        # TODO: this field might not be in v1. Double check later
        # StructField('session_id', StringType(), True),
        # StructField('user_id', StringType(), True),
    ])

    _preprocessed_data = [
        ["01", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 0, 3, 0), "OK", "2", "llm", "LLM", "in", "out"],
        ["01", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 0, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 0, 1, 0), "OK", "1", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 0, 12, 0), "[]", "[]", "name", "4"] +
        [datetime(2024, 2, 5, 0, 11, 0), "OK", "5", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 0, 7, 0), "OK", "4", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 0, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 0, 5, 0), "OK", "3", "llm", "LLM", "in", "out"],
    ]

    _preprocessed_schema_no_input = StructType([
        StructField('trace_id', StringType(), True),
        StructField('end_time', TimestampType(), True),
        StructField('events', StringType(), True),
        StructField('links', StringType(), True),
        StructField('name', StringType(), True),
        StructField('parent_id', StringType(), True),
        StructField('start_time', TimestampType(), True),
        StructField('status', StringType(), True),
        StructField('span_id', StringType(), True),
        StructField('span_type', StringType(), True),
        StructField('framework', StringType(), True),
        StructField('input', StringType(), True),
        StructField('output', StringType(), True),
        # TODO: this field might not be in v1. Double check later
        # StructField('session_id', StringType(), True),
        # StructField('user_id', StringType(), True),
    ])

    _preprocessed_data_no_inputs = [
        ["01", datetime(2024, 2, 7, 12, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 3, 0), "OK", "2", "llm", "LLM", None, "out"],
        ["01", datetime(2024, 2, 7, 12, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 1, 0), "OK", "1", "llm", "LLM", None, "out"],
        ["02", datetime(2024, 2, 7, 12, 12, 0), "[]", "[]", "name", "4"] +
        [datetime(2024, 2, 7, 12, 11, 0), "OK", "5", "llm", "LLM", None, "out"],
        ["02", datetime(2024, 2, 7, 12, 8, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 7, 0), "OK", "4", "llm", "LLM", None, "out"],
        ["02", datetime(2024, 2, 7, 12, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 5, 0), "OK", "3", "llm", "LLM", None, "out"],
    ]

    _preprocessed_data_look_back = [
        ["01", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 3, 0), "OK", "2", "llm", "LLM", "in", "out"],
        ["01", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 1, 0), "OK", "1", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 12, 0), "[]", "[]", "name", "4"] +
        [datetime(2024, 2, 10, 15, 11, 0), "OK", "5", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 5, 0), "OK", "3", "llm", "LLM", "in", "out"],
        ["02", "{\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 14, 10, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 14, 9, 0), "OK", "4", "llm", "LLM", "in", "out"],
    ]

    @pytest.mark.parametrize(
        "window_start_time, window_end_time, expected_schema, expected_data",
        [
            # data only
            (datetime(2024, 2, 5, 15), datetime(2024, 2, 5, 16), _preprocessed_schema, _preprocessed_data),
            # data and dataref mix
            # comment out the mix scenario due to package not found error from executor in remote run
            # (datetime(2024, 2, 20, 15), datetime(2024, 2, 20, 16), _preprocessed_schema, _preprocessed_data),
            # data but missing some promoted attribute fields
            (datetime(2024, 2, 7, 12), datetime(2024, 2, 7, 13),
             _preprocessed_schema_no_input, _preprocessed_data_no_inputs),
            # test the 1-hour lookback functionality
            (datetime(2024, 2, 10, 15), datetime(2024, 2, 10, 16),
             _preprocessed_schema, _preprocessed_data_look_back)
        ]
    )
    def test_genai_uri_folder_to_preprocessed_spark_df(
            self, genai_preprocessor_test_setup,
            window_start_time: datetime, window_end_time: datetime, expected_schema, expected_data):
        """Test uri_folder_to_spark_df()."""
        def my_add_tags(tags: dict):
            print("my_add_tags:", tags)

        print("testing mdc_uri_folder_to_preprocessed_spark_df...")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        input_url = StoreUrl(f"{tests_path}/unit/raw_genai_data/")

        actual_df = _genai_uri_folder_to_preprocessed_spark_df(
            window_start_time.strftime("%Y%m%dT%H:%M:%S"), window_end_time.strftime("%Y%m%dT%H:%M:%S"),
            input_url, my_add_tags)
        print("Preprocessed dataframe:")
        actual_df.show()
        actual_df.printSchema()

        spark = self._init_spark()
        expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

        print('Expected dataframe:')
        expected_df.show()
        expected_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_df)

    # def test_genai_preprocessor_fails(self, genai_preprocessor_test_setup):
    #     """Test scenarios where the preprocessor should throw validation errors."""
    #     def my_add_tags(tags: dict):
    #         print("my_add_tags:", tags)

    #     print("testing genai_uri_folder_to_preprocessed_spark_df...")
    #     tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
    #     input_url = StoreUrl(f"{tests_path}/unit/raw_genai_data/")

    #     # Data with invalid timestamps
    #     window_start_time = datetime(2024, 2, 8, 15)
    #     window_end_time = datetime(2024, 2, 8, 16)

    #     try:
    #         _ = _genai_uri_folder_to_preprocessed_spark_df(
    #             window_start_time.strftime("%Y%m%dT%H:%M:%S"), window_end_time.strftime("%Y%m%dT%H:%M:%S"),
    #             input_url, my_add_tags)
    #         assert False
    #     except Exception as ex:
    #         assert "The start or end time columns of the raw span logs contain invalid Timestamp strings." in str(ex)

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

    # @pytest.mark.parametrize(
    #         "span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema, require_trace_data",
    #         [
    #             ([], _preprocessed_log_schema, [], _trace_log_schema, True),
    #             (_span_log_data, _preprocessed_log_schema, _trace_log_data, _trace_log_schema, True),
    #             (
    #                 _span_log_data_extra, _preprocessed_log_schema_extra,
    #                 _trace_log_data_extra, _trace_log_schema, True),
    #             (_span_log_data, _preprocessed_log_schema, [], _trace_log_schema, False),
    #         ]
    # )
    # def test_trace_aggregator(
    #         self, genai_zip_test_setup, genai_preprocessor_test_setup,
    #         span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema, require_trace_data):
    #     """Test scenario where spans has real data."""
    #     spark = self._init_spark()
    #     # infer schema only when we have data.
    #     processed_spans_df = spark.createDataFrame(span_input_logs, span_input_schema)
    #     expected_traces_df = spark.createDataFrame(expected_trace_logs, expected_trace_schema)

    #     print("processed logs:")
    #     processed_spans_df.show()
    #     processed_spans_df.printSchema()

    #     print("expected trace logs:")
    #     expected_traces_df.show()
    #     expected_traces_df.printSchema()

    #     actual_trace_df = process_spans_into_aggregated_traces(processed_spans_df, require_trace_data)

    #     print("actual trace logs:")
    #     actual_trace_df.show()
    #     actual_trace_df.printSchema()

    #     assert_spark_dataframe_equal(actual_trace_df, expected_traces_df)


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    print(f'df1: {df1.collect()}')
    print(f'df2: {df2.collect()}')
    assert df1.collect() == df2.collect()
