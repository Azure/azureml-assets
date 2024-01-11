# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Joiner component."""

import pytest

from pyspark.sql.types import StringType, StructField, StructType
from src.model_monitor_data_joiner.run import join_data
from shared_utilities.momo_exceptions import InvalidInputError
from tests.e2e.utils.io_utils import create_pyspark_dataframe

LEFT_JOIN_COLUMN = 'left_join_column'
RIGHT_JOIN_COLUMN = 'right_join_column'
test_data = [(False, True), (True, False)]


def _generate_left_data_df(contains_join_column):
    left_data = [
        (3, 4, 1),
        (6, 7, 2),
        (9, 10, 3)
    ]
    left_columns = ['sepal_length', 'petal_length']

    if contains_join_column:
        left_columns.append(LEFT_JOIN_COLUMN)
    else:
        left_columns.append('random_column')

    return create_pyspark_dataframe(left_data, left_columns)


def _generate_right_data_df(contains_join_column):
    right_data = [
        ('flower1', 1),
        ('flower2', 2),
        ('flower3', 3)
    ]
    right_columns = ['target']

    if contains_join_column:
        right_columns.append(RIGHT_JOIN_COLUMN)
    else:
        right_columns.append('random_column')
    return create_pyspark_dataframe(right_data, right_columns)


@pytest.mark.unit1
class TestModelMonitorDataJoiner:
    """Test class for model monitor data joiner component."""

    def test_join_data_of_equal_length_successful(self):
        """Test join data to produce result successfully."""
        left_data_df = _generate_left_data_df(True)
        right_data_df = _generate_right_data_df(True)

        joined_data_df = join_data(
            left_data_df,
            LEFT_JOIN_COLUMN,
            right_data_df,
            RIGHT_JOIN_COLUMN
        )

        assert len(joined_data_df.columns) == 5
        assert joined_data_df.count() == 3

    @pytest.mark.parametrize("left_data_has_join_column, right_data_has_join_column", test_data)  # noqa
    def test_join_data_missing_join_column_raises_exception(
        self,
        left_data_has_join_column,
        right_data_has_join_column
    ):
        """Test join data to raises exception due to missing join column."""
        # Left does not contain join column
        left_data_df = _generate_left_data_df(left_data_has_join_column)
        right_data_df = _generate_right_data_df(right_data_has_join_column)

        with pytest.raises(Exception):
            join_data(
                left_data_df,
                LEFT_JOIN_COLUMN,
                right_data_df,
                RIGHT_JOIN_COLUMN
            )

    @pytest.mark.parametrize("is_left_data_empty, is_right_data_empty", test_data)
    def test_join_data_empty_input_successful(
        self,
        is_left_data_empty,
        is_right_data_empty
    ):
        """Test join data that produces empty result."""
        if is_left_data_empty:
            left_data_df = create_pyspark_dataframe(
                [],
                StructType([StructField(LEFT_JOIN_COLUMN, StringType(), True)])
            )
        else:
            left_data_df = _generate_left_data_df(True)

        if is_right_data_empty:
            right_data_df = create_pyspark_dataframe(
                [],
                StructType([StructField(RIGHT_JOIN_COLUMN, StringType(), True)])
            )
        else:
            right_data_df = _generate_right_data_df(True)

        with pytest.raises(InvalidInputError):
            join_data(
                left_data_df,
                LEFT_JOIN_COLUMN,
                right_data_df,
                RIGHT_JOIN_COLUMN
            )

    @pytest.mark.parametrize("is_left_data_empty, is_right_data_empty", test_data)
    def test_join_data_empty_input_without_schema_raises_exception(
        self,
        is_left_data_empty,
        is_right_data_empty
    ):
        """Test join data with empty input without schema."""
        if is_left_data_empty:
            left_data_df = create_pyspark_dataframe([], StructType([]))
        else:
            left_data_df = _generate_left_data_df(True)

        if is_right_data_empty:
            right_data_df = create_pyspark_dataframe([], StructType([]))
        else:
            right_data_df = _generate_right_data_df(True)

        with pytest.raises(InvalidInputError):
            join_data(
                left_data_df,
                LEFT_JOIN_COLUMN,
                right_data_df,
                RIGHT_JOIN_COLUMN
            )
