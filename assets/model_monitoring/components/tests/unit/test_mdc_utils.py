# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for MDC preprocessor helper."""

import pytest
from pyspark.sql import SparkSession
from datetime import datetime
from src.model_data_collector_preprocessor.mdc_utils import (
    _count_dropped_rows_with_error,
    _filter_df_by_time_window,
)
from tests.unit.utils.unit_test_utils import assert_spark_dataframe_equal


@pytest.mark.unit
class TestMDCUtils:
    """Test class for MDC utils file."""

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        spark: SparkSession = SparkSession.builder.appName("test").getOrCreate()
        return spark

    @pytest.mark.parametrize(
            "original_count,transformed_count,error_msg,expect_error",
            [
                (100, 100, "", False),
                (100, 95, "", False),
                (100, 30, "Test the error msg", True),
                (100, 120, "", False),
                (0, 1, "", False),
                (1, -3, "Dropped more rows than possible", True),
            ])
    def test_count_dropped_rows_with_error(
            self, original_count: int, transformed_count: int, error_msg: str, expect_error: bool):
        """Test."""
        if expect_error:
            try:
                _count_dropped_rows_with_error(original_count, transformed_count, error_msg)
                assert False
            except Exception as ex:
                assert error_msg in str(ex)
        else:
            _count_dropped_rows_with_error(original_count, transformed_count, error_msg)

    @pytest.mark.parametrize(
            "input_data, input_schema, expected_data, expected_schema",
            [
                # simple case. Include all data
                (
                    [(1, datetime(2024, 2, 3, 4, 10), datetime(2024, 2, 3, 4, 15)),
                     (2, datetime(2024, 2, 3, 4, 20), datetime(2024, 2, 3, 4, 25)),
                     (3, datetime(2024, 2, 3, 4, 30), datetime(2024, 2, 3, 4, 35)),
                     ],
                    ["order", "start_time", "end_time"],
                    [(1, datetime(2024, 2, 3, 4, 10), datetime(2024, 2, 3, 4, 15)),
                     (2, datetime(2024, 2, 3, 4, 20), datetime(2024, 2, 3, 4, 25)),
                     (3, datetime(2024, 2, 3, 4, 30), datetime(2024, 2, 3, 4, 35)),
                     ],
                    ["order", "start_time", "end_time"],
                ),
                # simple case. exclude some data
                (
                    [(0, datetime(2024, 2, 3, 3, 50), datetime(2024, 2, 3, 3, 59)),
                     (1, datetime(2024, 2, 3, 4, 10), datetime(2024, 2, 3, 4, 15)),
                     (2, datetime(2024, 2, 3, 4, 20), datetime(2024, 2, 3, 4, 25)),
                     (3, datetime(2024, 2, 3, 4, 30), datetime(2024, 2, 3, 4, 35)),
                     (4, datetime(2024, 2, 3, 4, 40), datetime(2024, 2, 3, 5)),
                     (5, datetime(2024, 2, 3, 4, 50), datetime(2024, 2, 3, 5, 1)),
                     ],
                    ["order", "start_time", "end_time"],
                    [(1, datetime(2024, 2, 3, 4, 10), datetime(2024, 2, 3, 4, 15)),
                     (2, datetime(2024, 2, 3, 4, 20), datetime(2024, 2, 3, 4, 25)),
                     (3, datetime(2024, 2, 3, 4, 30), datetime(2024, 2, 3, 4, 35)),
                     ],
                    ["order", "start_time", "end_time"],
                ),
                # Compare datetime to string and exclude some data
                (
                    [(0, "2024-02-03T03:50:00", "2024-02-03T03:59:00"),
                     (1, "2024-02-03T04:10:00", "2024-02-03T04:15:00"),
                     (2, "2024-02-03T04:20:00", "2024-02-03T04:25:00"),
                     (3, "2024-02-03T04:30:00", "2024-02-03T04:35:00"),
                     (4, "2024-02-03T04:40:00", "2024-02-03 05:00:00"),
                     (5, "2024-02-03T04:50:00", "2024-02-03 05:01:00"),
                     ],
                    ["order", "start_time", "end_time"],
                    [(1, "2024-02-03T04:10:00", "2024-02-03T04:15:00"),
                     (2, "2024-02-03T04:20:00", "2024-02-03T04:25:00"),
                     (3, "2024-02-03T04:30:00", "2024-02-03T04:35:00"),
                     ],
                    ["order", "start_time", "end_time"],
                ),
                # comparison works locally but not on github
                # Compare datetime to string w/ tzinfo and exclude some data
                # (
                #     [(0, "2024-02-03T11:50:00Z", "2024-02-03T11:59:00Z"),
                #      (1, "2024-02-03T12:10:00Z", "2024-02-03T12:15:00Z"),
                #      (2, "2024-02-03T12:20:00Z", "2024-02-03T12:25:00Z"),
                #      (3, "2024-02-03T12:30:00Z", "2024-02-03T12:35:00Z"),
                #      (4, "2024-02-03T12:40:00Z", "2024-02-03 13:00:00Z"),
                #      (5, "2024-02-03T12:50:00Z", "2024-02-03 13:01:00Z"),
                #      ],
                #     ["order", "start_time", "end_time"],
                #     [(1, "2024-02-03T12:10:00Z", "2024-02-03T12:15:00Z"),
                #      (2, "2024-02-03T12:20:00Z", "2024-02-03T12:25:00Z"),
                #      (3, "2024-02-03T12:30:00Z", "2024-02-03T12:35:00Z"),
                #      ],
                #     ["order", "start_time", "end_time"],
                # )
            ]
    )
    def test_filter_df_by_data_window(self, input_data, input_schema, expected_data, expected_schema):
        """Test scenarios for filtering dataframe by data window."""
        start_time = datetime(2024, 2, 3, 4)
        end_time = datetime(2024, 2, 3, 5)
        spark = self._init_spark()

        input_data_df = spark.createDataFrame(input_data, input_schema)
        expected_data_df = spark.createDataFrame(expected_data, expected_schema)

        actual_data_df = _filter_df_by_time_window(input_data_df, start_time, end_time)

        assert_spark_dataframe_equal(actual_data_df, expected_data_df)
