# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for GSQ - input schema adaptor component."""

import pytest
<<<<<<< HEAD
from pyspark.sql import DataFrame, SparkSession
=======
from pyspark.sql import SparkSession
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
from pyspark.sql.types import StructType, StructField, StringType
from src.generation_safety_quality.input_schema_adaptor.run import (
    _adapt_input_data_schema,
)
<<<<<<< HEAD
=======
from tests.unit.utils.unit_test_utils import assert_spark_dataframe_equal
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa


@pytest.mark.gsq_test
@pytest.mark.unit
class TestInputSchemaAdaptor:
    """Test class for GSQ - Input Schema Adaptor."""

    def _init_spark(self) -> SparkSession:
        """Create spark session."""
        return SparkSession.builder.appName("test").getOrCreate()

    _expected_gsq_input_schema = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("context", StringType(), True),
            StructField("groundtruth", StringType(), True),
            StructField("prompt", StringType(), True),
            StructField("output", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    _expected_gsq_input_schema_extra = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("context", StringType(), True),
            StructField("groundtruth", StringType(), True),
            StructField("prompt", StringType(), True),
            StructField("output", StringType(), True),
            StructField("source", StringType(), True),
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

    @pytest.mark.parametrize(
            "input_data, input_schema, expected_data, expected_schema",
            [
                # Test no data, should be pass-through
                ([], _simple_input_schema, [], _simple_input_schema),
                ([("", "")], _simple_input_schema, [("", "")], _simple_input_schema),
                # genai empty data
                ([], _genai_input_schema, [], _expected_gsq_input_schema_empty),
                # genai, empty input/output columns
                (
                    [("01", "null", "null", "null")], _genai_input_schema,
                    [("01", "null")], _expected_gsq_input_schema_empty
                ),
                # Test with genai columns
                (
                    [("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                      "{\"output\":\"answer\"}", "null")], _genai_input_schema,
                    [("01", "context", "ground-truth", "question", "answer", "null")], _expected_gsq_input_schema
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
                        ("01", "context", "ground-truth", "question", "answer", "LLM", "null"),
                    ], _expected_gsq_input_schema_extra
                ),
                # genai, data fall-through
                (
                    [("01", "{\"prompt\":\"question\",\"context\":\"context\",\"groundtruth\":\"ground-truth\"}",
                      "{\"output\":\"answer\",\"source\":\"LLM\"}", "null")], _genai_input_schema,
                    [("01", "context", "ground-truth", "question", "answer", "LLM", "null")],
                    _expected_gsq_input_schema_extra
                ),
            ]
    )
    def test_adapt_input_schema(self, input_data: list, input_schema: StructType,
                                expected_data: list, expected_schema: StructType):
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

        actual_df = _adapt_input_data_schema(input_data_df)
        print("Adapted schema dataframe:")
        actual_df.show()
        actual_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_data_df)

    def test_adapt_input_schema_bad_json(self):
        """Test scenario for input schema adaptor with invalid json object column."""
        spark = self._init_spark()
        input_data_df = spark.createDataFrame([("01", "asdfjkl", "421jjfd", "fdsa")], self._genai_input_schema)

        match_err = "Failed to parse the input/output column json string for the trace logs provided."
        try:
            _adapt_input_data_schema(input_data_df)
            pytest.fail("Should have thrown InvalidInputError exception.")
        except Exception as ex:
            assert match_err in str(ex)

    def test_adapt_input_schema_duplicate_columns(self):
        """Test scenario for input schema adaptor with invalid json object column."""
        spark = self._init_spark()
        input_data_df = spark.createDataFrame(
            [("01",
              "{\"prompt\":\"question\",\"context\":{\"name\": \"random\"},\"groundtruth\":\"ground-truth\"}",
              "{\"output\":\"answer\",\"context\":\"LLM\"}",
              "fdsa")],
            self._genai_input_schema)

        match_err = "Expanding the input and output columms resulted in duplicate columns."
        try:
            _adapt_input_data_schema(input_data_df)
            pytest.fail("Should have thrown InvalidInputError exception.")
        except Exception as ex:
            assert match_err in str(ex)
<<<<<<< HEAD


def assert_spark_dataframe_equal(actual_df: DataFrame, expected_df: DataFrame):
    """Assert two spark dataframes are equal."""
    assert actual_df.schema == expected_df.schema
    assert actual_df.count() == expected_df.count()
    assert actual_df.collect() == expected_df.collect()
=======
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
