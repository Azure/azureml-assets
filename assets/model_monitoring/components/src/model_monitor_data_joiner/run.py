# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Data Joiner Component."""

import argparse
from shared_utilities.io_utils import (
    read_mltable_in_spark,
    save_spark_df_as_mltable,
)
from shared_utilities.event_utils import post_warning_event


def join_data(
    left_input_data: str,
    left_join_column: str,
    right_input_data: str,
    right_join_column: str,
    joined_data: str
):
    """Join data assets based on the given join columns.

    Agrs:
        left_input_data: The left data asset to join.
        left_join_column: The join column for the left data asset.
        right_input_data: The right data asset to join.
        right_join_column: The join column for the right data asset.
    """
    # Load data
    left_input_data_df = read_mltable_in_spark(mltable_path=left_input_data)
    right_input_data_df = read_mltable_in_spark(mltable_path=right_input_data)

    # Join the data
    joined_data_df = left_input_data_df.join(
        right_input_data_df,
        left_input_data_df[left_join_column] == right_input_data_df[right_join_column],
        'inner'
    )

    # Post warning if the joined data
    if joined_data_df.count() == 0:
        error_message = "The joined data asset is empty. Either one or both the inputs to the component are empty or joining on 'join_column' resulted in empty result."
        post_warning_event(error_message)

    # Write the output
    save_spark_df_as_mltable(joined_data_df, joined_data)


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

    join_data(
        args.left_input_data,
        args.left_join_column,
        args.right_input_data,
        args.right_join_column,
        args.joined_data
    )
    print('Successfully executed data joiner component.')


if __name__ == "__main__":
    run()
