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
        # Collect All feature and importance value from input 
        feature_importance_names_importance = (
            self.feature_importance.select(self.feature_importance.feature, self.feature_importance.metric_value).collect()
        )
        # Create dictionary with feature as key and importance as value using row list 
        featureimportance_dictionary = {}
        for row in feature_importance_names_importance:
            featureimportance_dictionary[row[0]] = row[1]
        # Sort and find Top N features with higher importance
        top_N_feature_importance_names_importance = sorted(featureimportance_dictionary.items(),key=lambda x: x[1], reverse=True)[: self.N_value]
        # Get top N feature names and find common features in both input dataset
        top_features = [x[0] for x in top_N_feature_importance_names_importance]
        # Select top N common feature in both inputs
        return (
            FeatureSelectorAll()
            .select(input_df1, input_df2)
            .filter(col("featureName").isin(top_features))
        )
