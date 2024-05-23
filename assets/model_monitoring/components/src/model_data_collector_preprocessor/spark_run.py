# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse

from shared_utilities.io_utils import (
    save_spark_df_as_mltable, init_momo_component_environment,
)
from model_data_collector_preprocessor.mdc_utils import (
    _mdc_uri_folder_to_preprocessed_spark_df,
    _convert_complex_columns_to_json_string,
)
from shared_utilities.store_url import StoreUrl


def mdc_preprocessor(
        data_window_start: str,
        data_window_end: str,
        input_data: str,
        preprocessed_input_data: str,
        extract_correlation_id: bool):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
        extract_correlation_id: The boolean to extract correlation Id from the MDC logs.
    """
    store_url = StoreUrl(input_data)
    transformed_df = _mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url,
                                                              extract_correlation_id)
    # TODO remove this step after we switch our interface from mltable to uri_folder
    transformed_df = _convert_complex_columns_to_json_string(transformed_df)

    save_spark_df_as_mltable(transformed_df, preprocessed_input_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # setup momo environment
    init_momo_component_environment()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--extract_correlation_id", type=str)
    parser.add_argument("--preprocessed_input_data", type=str)
    args = parser.parse_args()

    mdc_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_input_data,
        eval(args.extract_correlation_id.capitalize()),
    )


if __name__ == "__main__":
    run()
