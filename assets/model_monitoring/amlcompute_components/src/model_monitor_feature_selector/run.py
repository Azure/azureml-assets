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
from shared_utilities.io_utils import read_mltable_in_spark, save_spark_df_as_mltable
from shared_utilities.patch_mltable import patch_all
patch_all()


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
        input_df1=read_mltable_in_spark(args.input_data_1),
        input_df2=read_mltable_in_spark(args.input_data_2),
    )

    save_spark_df_as_mltable(features_df, args.feature_names)


if __name__ == "__main__":
    run()
