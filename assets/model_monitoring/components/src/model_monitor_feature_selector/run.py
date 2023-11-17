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
from shared_utilities.io_utils import (
    try_read_mltable_in_spark, try_read_mltable_in_spark_with_error, save_spark_df_as_mltable
)
from shared_utilities.momo_exceptions import DataNotFoundError


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

    input_df1 = try_read_mltable_in_spark(args.input_data_1, "input_data_1")
    input_df2 = try_read_mltable_in_spark(args.input_data_2, "input_data_2")

    if not input_df1 and not input_df2:
        print("Both input data are empty. Skipping feature selection.")
        return
    elif not input_df1:
        input_df1 = input_df2
    # it's ok if input_df1 and not input_df2

    feature_importance = None
    try:
        feature_importance = try_read_mltable_in_spark_with_error(args.feature_importance, "feature_importance")
    except DataNotFoundError as e:
        if args.filter_type == FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name:
            raise e
    except Exception:
        if args.filter_type == FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name:
            raise Exception(
                "Error encountered when retrieving top N features. Please ensure target_column is defined."
            )

    feature_selector = FeatureSelectorFactory().produce(
        feature_selector_type=args.filter_type,
        filter_value=args.filter_value,
        feature_importance=feature_importance,
    )

    features_df = feature_selector.select(
        input_df1=input_df1,
        input_df2=input_df2,
    )

    save_spark_df_as_mltable(features_df, args.feature_names)


if __name__ == "__main__":
    run()
