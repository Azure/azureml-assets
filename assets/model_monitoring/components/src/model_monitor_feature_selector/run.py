# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse
from model_monitor_feature_selector.factories.feature_selector_factory import (
    FeatureSelectorFactory,
)
from model_monitor_feature_selector.selectors.feature_selector_type import (
    FeatureSelectorType,
)
from shared_utilities.io_utils import try_read_mltable_in_spark, read_mltable_in_spark, save_spark_df_as_mltable


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_1", type=str)
    parser.add_argument("--input_data_2", type=str, required=False, nargs="?")
    parser.add_argument("--filter_type", type=str)
    parser.add_argument("--filter_value", type=str)
    parser.add_argument("--feature_importance", type=str, required=False, nargs="?")
    parser.add_argument("--feature_names", type=str)
    args = parser.parse_args()

    input_df1 = try_read_mltable_in_spark(args.input_data_1, "input_data_1 is empty.")
    input_df2 = try_read_mltable_in_spark(args.input_data_2, "input_data_2 is empty.")

    if not input_df1 and not input_df2:
        raise ValueError("Both input data are empty. Cannot select the feature")

    elif not input_df1:
        input_df1 = input_df2

    feature_importance = None
    try:
        feature_importance = read_mltable_in_spark(args.feature_importance)
    except Exception:
        if args.filter_type == FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name:
            raise Exception(
                "Error encountered when retrieving top N features. Please ensure target_column is defined."
            )

    factory = FeatureSelectorFactory().produce(
        feature_selector_type=args.filter_type,
        filter_value=args.filter_value,
        feature_importance=feature_importance,
    )

    features_df = factory.select(
        input_df1=input_df1,
        input_df2=input_df2,
    )

    if features_df is None or features_df.isEmpty():
         raise ValueError("Could not find common column names.")       
    
    save_spark_df_as_mltable(features_df, args.feature_names)


if __name__ == "__main__":
    run()
