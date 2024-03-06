# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Strategy class which selects all features."""
import pyspark.sql as pyspark_sql


class FeatureSelector:
    """Builder class which creates a feature selector."""

    def select(self, input_df1: pyspark_sql.DataFrame, input_df2: pyspark_sql.DataFrame) -> pyspark_sql.DataFrame:
        """Select all common features between both input data frames."""
        return
