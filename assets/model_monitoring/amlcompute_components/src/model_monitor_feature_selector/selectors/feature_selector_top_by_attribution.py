# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strategy class which selects top n features by feature importance."""

import pyspark.sql as pyspark_sql
from pyspark.sql.functions import col
from model_monitor_feature_selector.selectors.feature_selector import FeatureSelector
from model_monitor_feature_selector.selectors.feature_selector_all import (
    FeatureSelectorAll,
)


class FeatureSelectorTopNByAttribution(FeatureSelector):
    """Builder class which creates a Prediction Drift signal output."""

    def __init__(self, feature_importance: pyspark_sql.DataFrame, filter_value):
        """Construct a FeatureMetric instance."""
        self.feature_importance = feature_importance.filter(col("feature") != "")
        self.N_value = filter_value

    def select(
        self, input_df1: pyspark_sql.DataFrame, input_df2: pyspark_sql.DataFrame
    ) -> pyspark_sql.DataFrame:
        """Select the top N contributing features."""
        feature_importance_names = (
            self.feature_importance.select("feature").rdd.flatMap(lambda x: x).collect()
        )
        top_N_feature_importance_names = feature_importance_names[: self.N_value]
        return (
            FeatureSelectorAll()
            .select(input_df1, input_df2)
            .filter(col("featureName").isin(top_N_feature_importance_names))
        )
