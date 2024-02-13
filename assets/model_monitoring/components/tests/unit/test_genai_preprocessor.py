# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for Gen AI preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField, StringType, DoubleType, BooleanType, IntegerType, LongType, TimestampType,
    ArrayType, StructType, MapType
)
import pytest
import os
import sys
import json
from datetime import datetime
from src.model_data_collector_preprocessor.genai_run import (
    _preprocess_raw_logs_to_span_logs_spark_df,
    _get_preprocessed_span_logs_df_schema,
    _mdc_uri_folder_to_preprocessed_spark_df
)
from src.model_data_collector_preprocessor.store_url import StoreUrl
from src.shared_utilities.momo_exceptions import DataNotFoundError


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
class TestGenAISparkPreprocessor:
    """Test class for Gen AI Preprocessor."""

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        return SparkSession.builder.appName("test").getOrCreate()

    _preprocessed_schema = StructType([
        StructField('trace_id', StringType(), True),
        StructField('span_id', StringType(), True),
        StructField('parent_id', StringType(), True),
        # TODO: this field might not be in v1. Double check later
        # StructField('user_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_type', StringType(), True),
        StructField('start_time', TimestampType(), True),
        StructField('end_time', TimestampType(), True),
        StructField('name', StringType(), True),
        StructField('status', StringType(), True),
        StructField('framework', StringType(), True),
        StructField('input', StringType(), True),
        StructField('output', StringType(), True),
        StructField('attributes', StringType(), True),
        StructField('events', StringType(), True),
        StructField('links', StringType(), True),
    ])
    _preprocessed_data = [
        ["01", "1", None, "llm", datetime(2024,2,5,0,1,0)] + \
            [datetime(2024,2,5,0,2,0), "name", "OK", "LLM", "in", "out", "{}", "[]", "[]"],
        ["01", "2", None, "llm", datetime(2024,2,5,0,3,0)] + \
            [datetime(2024,2,5,0,4,0), "name", "OK", "LLM", "in", "out", "{}", "[]", "[]"],
        ["02", "3", None, "llm", datetime(2024,2,5,0,5,0)] + \
            [datetime(2024,2,5,0,6,0), "name", "OK", "LLM", "in", "out", "{}", "[]", "[]"],
        ["02", "4", None, "llm", datetime(2024,2,5,0,7,0)] + \
            [datetime(2024,2,5,0,8,0), "name", "OK", "LLM", "in", "out", "{}", "[]", "[]"],
    ]

    @pytest.mark.parametrize(
        "window_start_time, window_end_time, expected_schema, expected_data",
        [
            # data only
            (datetime(2024, 2, 5, 15), datetime(2024, 2, 5, 16), _preprocessed_schema, _preprocessed_data),
            # data and dataref mix
            # comment out the mix scenario due to package not found error from executor in remote run
            # (datetime(2023, 10, 15, 17), datetime(2023, 10, 15, 18), _preprocessed_schema, _preprocessed_data),
            # dataref only
            # dataref only is not supported yet due to lack of schema
            # (datetime(2023, 10, 16, 21), datetime(2023, 10, 16, 22), _preprocessed_schema, _preprocessed_data),
        ]
    )
    def test_mdc_uri_folder_to_preprocessed_spark_df(
            self, genai_preprocessor_test_setup, window_start_time: datetime, window_end_time: datetime,
            expected_schema, expected_data):
        """Test uri_folder_to_spark_df()."""
        def my_add_tags(tags: dict):
            print("my_add_tags:", tags)

        print("testing mdc_uri_folder_to_preprocessed_spark_df...")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        input_url = StoreUrl(f"{tests_path}/unit/raw_genai_data/")

        actual_df = _mdc_uri_folder_to_preprocessed_spark_df(
            window_start_time.strftime("%Y%m%dT%H:%M:%S"), window_end_time.strftime("%Y%m%dT%H:%M:%S"),
            input_url, False, my_add_tags)
        print("raw dataframe:")
        actual_df.show()
        actual_df.printSchema()

        print("preprocessed span logs:")
        actual_df = _preprocess_raw_logs_to_span_logs_spark_df(actual_df)
        actual_df.show()
        actual_df.printSchema()

        print('expected data:')
        spark = self._init_spark()
        expected_df = spark.createDataFrame(expected_data, schema=expected_schema)

        expected_df.show()
        expected_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_df)

def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    print(f'df1: {df1.collect()}')
    print(f'df2: {df2.collect()}')
    assert df1.collect() == df2.collect()
