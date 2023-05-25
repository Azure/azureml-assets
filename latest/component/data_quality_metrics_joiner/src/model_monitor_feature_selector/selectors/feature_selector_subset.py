# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strategy class which selects all features."""

import pyspark.sql as pyspark_sql
from pyspark.sql.functions import col
from model_monitor_feature_selector.selectors.feature_selector import FeatureSelector
from model_monitor_feature_selector.selectors.feature_selector_all import (
    FeatureSelectorAll,
)


class FeatureSelectorSubset(FeatureSelector):
    """Builder class which creates a Prediction Drift signal output."""

    def __init__(self, feature_names: list):
        """Construct a FeatureMetric instance."""
        self.feature_names = feature_names

    def select(
        self, input_df1: pyspark_sql.DataFrame, input_df2: pyspark_sql.DataFrame
    ) -> pyspark_sql.DataFrame:
        """Select a subset of common features between both input data frames."""
        return (
            FeatureSelectorAll()
            .select(input_df1, input_df2)
            .where(col("featureName").isin(self.feature_names))
        )
