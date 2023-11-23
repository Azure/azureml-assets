# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Data Joiner Component."""

import argparse
import pyspark.sql as pyspark_sql
from shared_utilities.event_utils import post_warning_event
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
)


def _validate_join_column_in_input_data(
    input_data_df: pyspark_sql.DataFrame,
    join_column: str,
    input_data_name: str
):
    if join_column not in input_data_df.columns:
        raise Exception(f"The join column '{join_column}' is not present in {input_data_name}.")


def join_data(
    left_input_data_df: pyspark_sql.DataFrame,
    left_join_column: str,
    right_input_data_df: pyspark_sql.DataFrame,
    right_join_column: str
) -> pyspark_sql.DataFrame:
    """Join data assets based on the given join columns.

    Agrs:
        left_input_data_df: The dataframe for the left input data to join.
        left_join_column: The join column for the left data asset.
        right_input_data_df: The dataframe for the right input data to join.
        right_join_column: The join column for the right data asset.
    """
    # Validate
    _validate_join_column_in_input_data(left_input_data_df, left_join_column, "left_input_data")
    _validate_join_column_in_input_data(right_input_data_df, right_join_column, "right_input_data")

    # Join the data
    if left_join_column == right_join_column:
        joined_data_df = left_input_data_df.join(
            right_input_data_df,
            left_input_data_df[left_join_column] == right_input_data_df[right_join_column],
            'inner'
        ).drop(right_input_data_df[right_join_column])
    else:
        joined_data_df = left_input_data_df.join(
            right_input_data_df,
            left_input_data_df[left_join_column] == right_input_data_df[right_join_column],
            'inner'
        )

    return joined_data_df


def run():
    """Join data assets on given columns."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--left_input_data", type=str, required=True)
    parser.add_argument("--left_join_column", type=str, required=True)
    parser.add_argument("--right_input_data", type=str, required=True)
    parser.add_argument("--right_join_column", type=str, required=True)
    parser.add_argument("--joined_data", type=str, required=True)
    args = parser.parse_args()

    # Load data
    left_input_data_df = try_read_mltable_in_spark_with_error(args.left_input_data, "left_input_data")
    right_input_data_df = try_read_mltable_in_spark_with_error(args.right_input_data, "right_input_data")

    joined_data_df = join_data(
        left_input_data_df,
        args.left_join_column,
        right_input_data_df,
        args.right_join_column
    )

    # Raise warning if the result is empty
    if joined_data_df.count() == 0:
        warning_message = 'The data joiner resulted in an empty data asset. Please check the input data to see if this is expected.'  # noqa
        post_warning_event(warning_message)

    # Write the joined data.
    save_spark_df_as_mltable(joined_data_df, args.joined_data)
    print('Successfully executed data joiner component.')


if __name__ == "__main__":
    run()
