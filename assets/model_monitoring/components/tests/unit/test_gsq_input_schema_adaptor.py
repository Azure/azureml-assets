# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for GSQ - input schema adaptor component."""

import os
import pytest
import sys
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from src.generation_safety_quality.input_schema_adaptor.run import (
    _adapt_input_data_schema,
)
import spark_mltable  # noqa, to enable spark.read.mltable
from spark_mltable import SPARK_ZIP_PATH


@pytest.fixture(scope="module")
def gsq_preprocessor_test_setup():
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

@pytest.mark.gsq_test
@pytest.mark.unit
class TestInputSchemaAdaptor:
    """Test class for GSQ - Input Schema Adaptor."""

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        spark: SparkSession = SparkSession.builder.appName("test").getOrCreate()
        sc = spark.sparkContext
        # if SPARK_ZIP_PATH is set, add the zip file to the spark context
        zip_path = os.environ.get(SPARK_ZIP_PATH, '')
        print(f"The spark_zip in environment: {zip_path}")
        if zip_path:
            sc.addPyFile(zip_path)
        return spark

    _expected_gsq_input_schema = StructType(
        [
            StructField("prompt", StringType(), True),
            StructField("output", StringType(), True),
            StructField("context", StringType(), True),
            StructField("groundtruth", StringType(), True),
            StructField("trace_id", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    _expected_gsq_input_schema_drop_null = StructType(
        [
            StructField("prompt", StringType(), True),
            StructField("output", StringType(), True),
            StructField("context", StringType(), True),
            StructField("trace_id", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    _expected_gsq_input_schema_empty = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    _simple_input_schema = StructType(
        [
            StructField("input", StringType(), True),
            StructField("output", StringType(), True),
        ]
    )

    _genai_input_schema = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("input", StringType(), True),
            StructField("output", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    _default_column_mapping = {
        "prompt_mapping": "prompt",
        "completion_mapping": "output",
        "context_mapping": "context",
        "groundtruth_mapping": "groundtruth"
    }

    @pytest.mark.parametrize(
            "input_data, input_schema, expected_data, expected_schema, column_mappings",
            [
                # # Test no data, should be pass-through
                # ([], _simple_input_schema, [], _simple_input_schema, _default_column_mapping),
                # ([("", "")], _simple_input_schema, [("", "")], _simple_input_schema, _default_column_mapping),
                # # genai empty data
                # ([], _genai_input_schema, [], _expected_gsq_input_schema_empty, _default_column_mapping),
                # genai, empty input/output columns
                (
                    [("01", "null", "null", "null")], _genai_input_schema,
                    [("01", "null")], _expected_gsq_input_schema_empty,
                    _default_column_mapping
                ),
                # Test with genai columns. Default column mapping
                (
                    [("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                      "{\"output\":\"answer\"}", "null")], _genai_input_schema,
                    [("01", "context", "ground-truth", "question", "answer", "null")], _expected_gsq_input_schema,
                    _default_column_mapping
                ),
                # Test with genai columns. Mismatched json schema on rows
                (
                    [
                        ("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                         "{\"output\":\"answer\"}", "null"),
                        ("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                         "{\"output\":\"answer\", \"source\":\"LLM\"}", "null")
                      ], _genai_input_schema,
                    [
                        ("01", "context", "ground-truth", "question", "answer", None, "null"),
                        ("01", "context", "ground-truth", "question", "answer", "null"),
                    ], _expected_gsq_input_schema_drop_null, _default_column_mapping
                ),
                # genai, data fall-through
                (
                    [("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                      "{\"output\":\"answer\",\"source\":\"LLM\"}", "null")], _genai_input_schema,
                    [("01", "context", "ground-truth", "question", "answer", "null")],
                    _expected_gsq_input_schema_drop_null, _default_column_mapping
                ),
            ]
    )
    def test_adapt_input_schema(self, code_zip_test_setup, gsq_preprocessor_test_setup,
                                input_data: list, input_schema: StructType,
                                expected_data: list, expected_schema: StructType,
                                column_mappings: dict):
        """Test scenario happy path for input schema adaptor."""
        spark = self._init_spark()
        input_data_df = spark.createDataFrame(input_data, input_schema)
        expected_data_df = spark.createDataFrame(expected_data, expected_schema)

        print("Input dataframe:")
        input_data_df.show()
        input_data_df.printSchema()

        print("Expected dataframe:")
        expected_data_df.show()
        expected_data_df.printSchema()

        prompt = column_mappings.get("prompt_mapping")
        completion = column_mappings.get("completion_mapping")
        context = column_mappings.get("context_mapping")
        ground_truth = column_mappings.get("groundtruth_mapping")
        actual_df = _adapt_input_data_schema(input_data_df, prompt, completion, context, ground_truth)
        print("Adapted schema dataframe:")
        actual_df.show()
        actual_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_data_df)

    # def test_adapt_input_schema_bad_json(self):
    #     """Test scenario for input schema adaptor with invalid json object column."""
    #     spark = self._init_spark()
    #     input_data_df = spark.createDataFrame([("01", "asdfjkl", "421jjfd", "fdsa")], self._genai_input_schema)

    #     match_err = "Failed to parse the input/output column json string for the trace logs provided."
    #     try:
    #         _adapt_input_data_schema(input_data_df, "", "", "", "")
    #         pytest.fail("Should have thrown InvalidInputError exception.")
    #     except Exception as ex:
    #         assert match_err in str(ex)

    # def test_adapt_input_schema_duplicate_columns(self):
    #     """Test scenario for input schema adaptor with invalid json object column."""
    #     spark = self._init_spark()
    #     input_data_df = spark.createDataFrame(
    #         [("01",
    #           "{\"prompt\":\"question\",\"context\":{\"name\": \"random\"},\"groundtruth\":\"ground-truth\"}",
    #           "{\"output\":\"answer\",\"context\":\"LLM\"}",
    #           "fdsa")],
    #         self._genai_input_schema)

    #     match_err = "Expanding the input and output columms resulted in duplicate columns."
    #     try:
    #         _adapt_input_data_schema(input_data_df)
    #         pytest.fail("Should have thrown InvalidInputError exception.")
    #     except Exception as ex:
    #         assert match_err in str(ex)


def assert_spark_dataframe_equal(actual_df: DataFrame, expected_df: DataFrame):
    """Assert two spark dataframes are equal."""
    assert actual_df.schema == expected_df.schema
    assert actual_df.count() == expected_df.count()
    assert actual_df.collect() == expected_df.collect()
