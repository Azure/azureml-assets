# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the FeatureSelectorAll class."""

import pytest
from pyspark.sql.types import StructType, StructField, DoubleType, FloatType
from src.model_monitor_feature_selector.selectors.feature_selector_all import FeatureSelectorAll
from tests.e2e.utils.io_utils import create_pyspark_dataframe


@pytest.mark.unit
class TestFeatureSelectorAll:
    """Test class for feature selector component."""

    def test_feature_selector_all_select_expect_succeed(self):
        """Test feature selector scenarios."""
        feature_selector = FeatureSelectorAll()

        # Test with two dataframes that have common columns but datatype is not in the same type
        float_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", FloatType(), True)])
        baseline_df = create_pyspark_dataframe(float_data, schema)
        double_data = [(3.55,), (6.88,), (7.99,)]
        schema = StructType([
            StructField("target", DoubleType(), True)])
        production_df = create_pyspark_dataframe(double_data, schema)
        features = feature_selector.select(baseline_df, production_df)
        assert features.count() == 1

    def test_feature_selector_all_select_no_common_columns_expect_failure(self):
        """Test feature selector scenarios."""
        feature_selector = FeatureSelectorAll()

        # Test with two dataframes that have no common columns
        baseline_df = create_pyspark_dataframe([(1, "a"), (2, "b")],
                                               ["id", "name"])
        production_df = create_pyspark_dataframe([(3, "c"), (4, "d")],
                                                 ["age", "gender"])
        try:
            feature_selector.select(baseline_df, production_df)
        except Exception as e:
            assert "Found no common columns between input datasets." in e.args[0]
