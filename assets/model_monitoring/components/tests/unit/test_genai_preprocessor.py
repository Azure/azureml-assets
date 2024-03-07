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
from datetime import datetime
from model_data_collector_preprocessor.genai_run import (
    _genai_uri_folder_to_preprocessed_spark_df,
)
from model_data_collector_preprocessor.store_url import StoreUrl
from shared_utilities.io_utils import init_spark
import spark_mltable  # noqa, to enable spark.read.mltable

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

    _preprocessed_schema = StructType([
        StructField('attributes', StringType(), True),
        StructField('context', StringType(), True),
        StructField('end_time', TimestampType(), True),
        StructField('events', StringType(), True),
        StructField('links', StringType(), True),
        StructField('name', StringType(), True),
        StructField('parent_id', StringType(), True),
        StructField('start_time', TimestampType(), True),
        StructField('status', StringType(), True),
        StructField('trace_id', StringType(), True),
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
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"1\",\"trace_id\":\"01\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 15, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 15, 1, 0), "OK", "01", "1", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"2\",\"trace_id\":\"01\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 15, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 15, 3, 0), "OK", "01", "2", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"3\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 15, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 15, 5, 0), "OK", "02", "3", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"4\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 15, 8, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 5, 15, 7, 0), "OK", "02", "4", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"5\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 5, 15, 12, 0), "[]", "[]", "name", "4"] +
        [datetime(2024, 2, 5, 15, 11, 0), "OK", "02", "5", "llm", "LLM", "in", "out"],
    ]

    _preprocessed_schema_no_input = StructType([
        StructField('attributes', StringType(), True),
        StructField('context', StringType(), True),
        StructField('end_time', TimestampType(), True),
        StructField('events', StringType(), True),
        StructField('links', StringType(), True),
        StructField('name', StringType(), True),
        StructField('parent_id', StringType(), True),
        StructField('start_time', TimestampType(), True),
        StructField('status', StringType(), True),
        StructField('trace_id', StringType(), True),
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
        ["{\"framework\":\"LLM\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"1\",\"trace_id\":\"01\"}"] +
        [datetime(2024, 2, 7, 12, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 1, 0), "OK", "01", "1", "llm", "LLM", None, "out"],
        ["{\"framework\":\"LLM\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"2\",\"trace_id\":\"01\"}"] +
        [datetime(2024, 2, 7, 12, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 3, 0), "OK", "01", "2", "llm", "LLM", None, "out"],
        ["{\"framework\":\"LLM\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"3\",\"trace_id\":\"02\"}"] +
        [datetime(2024, 2, 7, 12, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 5, 0), "OK", "02", "3", "llm", "LLM", None, "out"],
        ["{\"framework\":\"LLM\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"4\",\"trace_id\":\"02\"}"] +
        [datetime(2024, 2, 7, 12, 8, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 7, 12, 7, 0), "OK", "02", "4", "llm", "LLM", None, "out"],
        ["{\"framework\":\"LLM\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"5\",\"trace_id\":\"02\"}"] +
        [datetime(2024, 2, 7, 12, 12, 0), "[]", "[]", "name", "4"] +
        [datetime(2024, 2, 7, 12, 11, 0), "OK", "02", "5", "llm", "LLM", None, "out"],
    ]

    _preprocessed_data_look_back = [
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"1\",\"trace_id\":\"01\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 2, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 1, 0), "OK", "01", "1", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"2\",\"trace_id\":\"01\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 4, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 3, 0), "OK", "01", "2", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"3\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 6, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 5, 0), "OK", "02", "3", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"4\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 15, 12, 0), "[]", "[]", "name", "5"] +
        [datetime(2024, 2, 10, 15, 11, 0), "OK", "02", "4", "llm", "LLM", "in", "out"],
        ["{\"framework\":\"LLM\",\"inputs\":\"in\",\"output\":\"out\",\"span_type\":\"llm\"}"] +
        ["{\"span_id\":\"5\",\"trace_id\":\"02\",\"trace_state\":\"[]\"}"] +
        [datetime(2024, 2, 10, 16, 1, 0), "[]", "[]", "name", None] +
        [datetime(2024, 2, 10, 15, 11, 0), "OK", "02", "5", "llm", "LLM", "in", "out"],
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
            # TODO: uncomment when support 1-hour look back/forward
            # test the 1-hour lookback functionality
            # (datetime(2024, 2, 10, 15), datetime(2024, 2, 10, 16),
            #  _preprocessed_schema, _preprocessed_data_look_back)
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

        spark = init_spark()
        expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

        print('Expected dataframe:')
        expected_df.show()
        expected_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_df)

    def test_genai_preprocessor_fails(self, genai_preprocessor_test_setup):
        """Test scenarios where the preprocessor should throw validation errors."""
        def my_add_tags(tags: dict):
            print("my_add_tags:", tags)

        print("testing genai_uri_folder_to_preprocessed_spark_df...")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        input_url = StoreUrl(f"{tests_path}/unit/raw_genai_data/")

        # Data with invalid timestamps
        window_start_time = datetime(2024, 2, 8, 15)
        window_end_time = datetime(2024, 2, 8, 16)

        actual_data = _genai_uri_folder_to_preprocessed_spark_df(
            window_start_time.strftime("%Y%m%dT%H:%M:%S"), window_end_time.strftime("%Y%m%dT%H:%M:%S"),
            input_url, my_add_tags)
        assert actual_data.isEmpty()


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    print(f'df1: {df1.collect()}')
    print(f'df2: {df2.collect()}')
    assert df1.collect() == df2.collect()
