# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strategy class which selects all features."""
import pyspark.sql as pyspark_sql


class FeatureSelector:
    """Builder class which creates a Prediction Drift signal output."""

    def select(self, input_df: pyspark_sql.DataFrame) -> pyspark_sql.DataFrame:
        """Select all common features between both input data frames."""
        return
