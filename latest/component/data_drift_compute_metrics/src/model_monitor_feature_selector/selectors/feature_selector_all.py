# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strategy class which selects all features."""

from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
)
from pyspark.sql import Row

import pyspark.sql as pyspark_sql
from model_monitor_feature_selector.selectors.feature_selector import FeatureSelector
from shared_utilities.io_utils import init_spark


class FeatureSelectorAll(FeatureSelector):
    """Builder class which creates a Prediction Drift signal output."""

    def select(
        self, input_df1: pyspark_sql.DataFrame, input_df2: pyspark_sql.DataFrame
    ) -> pyspark_sql.DataFrame:
        """Select all common features between both input data frames."""
        # Get the schema of the DataFrame
        schema = StructType(
            [
                StructField("featureName", StringType(), True),
            ]
        )

        rows = []
        df1_cols = input_df1.columns

        df2_cols = df1_cols
        if input_df2 is not None:
            df2_cols = input_df2.columns

        for feature in df1_cols:
            if feature in df2_cols:
                rows.append(Row(feature))

        spark = init_spark()
        features = spark.createDataFrame(data=rows, schema=schema)
        features.show()
        if features.isEmpty():
            raise Exception(
                "Could not generate features set correctly. Found no common columns between input datasets."
            )

        return features
