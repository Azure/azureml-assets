# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

from feature_importance_metrics.compute_feature_importance import determine_task_type, get_train_test_data
from feature_importance_metrics.feature_importance_utilities import mark_categorical_column
from feature_importance_metrics.compute_feature_attribution_drift import (
    drop_metadata_columns, calculate_attribution_drift)
import pytest
import pandas as pd


@pytest.fixture
def get_fraud_data():
    """Return fraud data as pandas dataframe."""
    return pd.DataFrame({
        "TRANSACTIONID": ["6175BE16-6602-4B7E-8225-D09BC4BCB59D",
                          "2EBBE4D6-4527-473F-B5E9-D946C4FD18E1",
                          "7B1FB72A-69D0-4A2A-A683-E451D470DBEB",
                          "946E2DF1-2DB6-43F4-AF26-790BFDCD2C78"],
        "ACCOUNTID": ["A1176337474875483", "A1343835256155075",
                      "A1343835256155076", "A1706480214256418"],
        "TRANSACTIONAMOUNT": [146161.99, 57487.200000000004,  227728.76, 59340.0],
        "TIMESTAMPSTR": ["2023-01-30T16:25:17.000Z",
                         "2023-01-30T16:58:44.000Z",
                         "2023-01-30T22:46:37.000Z",
                         "2023-01-30T22:46:37.000Z"],
        "TIMESTAMP": [pd.Timestamp('2023-12-04T17:59:32'),
                      pd.Timestamp('2023-12-04T17:59:32'),
                      pd.Timestamp('2023-12-04T17:59:32'),
                      pd.Timestamp('2023-12-04T17:59:32')],
        "TRANSACTIONAMOUNTUSD": [169547.9084, 57487.200000000004, 266442.6492, 67647.6],
        "ISPROXYIP": [False, True, False, False],
        "DIGITALITEMCOUNT": [15, 15, 15, 15],
        "PHYSICALITEMCOUNT": [0, 1, 1, 0],
        "IS_FRAUD": ["0", "0", "0", "0"]
    })


@pytest.fixture
def get_zipcode_data():
    """Return zipcode data as pandas dataframe."""
    return pd.DataFrame({
            "zipcode": [10001] * 21,
            "location": ["Seattle"] * 21
        })


@pytest.fixture
def get_large_data():
    """Return large zipcode data as pandas dataframe."""
    return pd.DataFrame({
            "zipcode": [10001] * 10003,
            "location": ["Seattle"] * 10003
        })


@pytest.mark.unit
class TestComputeFeatureImportanceMetrics:
    """Test class for feature importance component and utilities."""

    def test_get_train_test_data(self, get_large_data):
        """Test split data ."""
        train_data, test_data = get_train_test_data(get_large_data)
        assert len(train_data.index) == 5003
        assert len(test_data.index) == 5000

    def test_mark_categorical_column(self, get_fraud_data):
        """Test deteremine task type for classification scenario."""
        categorical_features_lgbm = ["TRANSACTIONID", "ACCOUNTID", "TIMESTAMPSTR", "TIMESTAMP", "ISPROXYIP"]
        numerical_features = ["TRANSACTIONAMOUNT", "TRANSACTIONAMOUNTUSD", "DIGITALITEMCOUNT"]
        target_column = "IS_FRAUD"

        mark_categorical_column(get_fraud_data, target_column, categorical_features_lgbm, numerical_features)
        # timestamp/date should be mark as int
        assert get_fraud_data.dtypes['TIMESTAMP'] == "int64"
        # categorical columns should mark as category
        assert get_fraud_data.dtypes['TIMESTAMPSTR'] == "category"
        assert get_fraud_data.dtypes['TRANSACTIONID'] == "category"
        assert get_fraud_data.dtypes['ACCOUNTID'] == "category"
        assert get_fraud_data.dtypes['ISPROXYIP'] == "category"
        # non-categorical columns should not mark as category
        assert get_fraud_data.dtypes['TRANSACTIONAMOUNT'] == "float64"
        assert get_fraud_data.dtypes['TRANSACTIONAMOUNTUSD'] == "float64"
        assert get_fraud_data.dtypes['DIGITALITEMCOUNT'] == "int64"
        # target column should not mark as category
        assert get_fraud_data.dtypes['IS_FRAUD'] == "object"
        # Unknown type should mark as category
        assert get_fraud_data.dtypes['PHYSICALITEMCOUNT'] == "category"

    def test_determine_task_type_classification(self, get_fraud_data, get_zipcode_data):
        """Test deteremine task type for classification scenario."""
        categorical_features = ["TRANSACTIONID", "ACCOUNTID", "TIMESTAMP", "ISPROXYIP"]
        task_type = determine_task_type(None, "ISPROXYIP", get_fraud_data, categorical_features)
        assert task_type == "classification"
        task_type = determine_task_type("invalid", "TRANSACTIONID", get_zipcode_data, categorical_features)
        assert task_type == "classification"
        task_type = determine_task_type("Classification", "zipcode", get_zipcode_data, categorical_features)
        assert task_type == "classification"
        task_type = determine_task_type("Classification", "TIMESTAMP", get_fraud_data, categorical_features)
        assert task_type == "classification"

    def test_determine_task_type_regression(self, get_fraud_data):
        """Test deteremine task type for regression scenario."""
        categorical_features = ["TRANSACTIONID", "ACCOUNTID", "TIMESTAMP", "ISPROXYIP"]
        task_type = determine_task_type(None, "TRANSACTIONAMOUNTUSD", get_fraud_data, categorical_features)
        assert task_type == "regression"
        task_type = determine_task_type("invalid", "TRANSACTIONAMOUNTUSD", get_fraud_data, categorical_features)
        assert task_type == "regression"
        task_type = determine_task_type("Regression", "TRANSACTIONAMOUNTUSD", get_fraud_data, categorical_features)
        assert task_type == "regression"

    def test_drop_metadata_columns(self):
        """Test drop columns when baseline and production do not match."""
        baseline_data = {
            "feature": ["col1", "col2", "col3"],
            "metric_name": ["FeatureImportance", "FeatureImportance", "FeatureImportance"],
            "metric_value": [.1, .4, .5]
        }
        baseline_dataframe = pd.DataFrame(baseline_data)
        production_data = {
            "feature": ["col1", "col2", "col4"],
            "metric_name": ["FeatureImportance", "FeatureImportance", "FeatureImportance"],
            "metric_value": [.1, .4, .5],
        }
        production_dataframe = pd.DataFrame(production_data)
        result = drop_metadata_columns(baseline_dataframe, production_dataframe)
        assert result["feature"].equals(pd.Series(["col1", "col2"]))

    def test_calculated_attribution_no_drift(self):
        """Test calculating ndcg metric with no drift."""
        baseline_data = {
            "feature": ["col1", "col2", "col3", ""],
            "metric_name": ["FeatureImportance", "FeatureImportance", "FeatureImportance", "RowCount"],
            "metric_value": [.1, .4, .5, 3]
        }
        baseline_dataframe = pd.DataFrame(baseline_data)
        production_data = {
            "feature": ["col1", "col2", "col3", ""],
            "metric_name": ["FeatureImportance", "FeatureImportance", "FeatureImportance", "RowCount"],
            "metric_value": [.1, .4, .5, 3],
        }
        production_dataframe = pd.DataFrame(production_data)
        drift = calculate_attribution_drift(baseline_dataframe, production_dataframe)
        assert drift == 1.0

    def test_calculated_attribution_with_drift(self):
        """Test calculating ndcg metric with drift."""
        baseline_data = {
            "feature": ["col1", "col2", "col3", ""],
            "metric_name": ["FeatureImportance", "FeatureImportance", "FeatureImportance", "RowCount"],
            "metric_value": [1.2, 0, .1, 3]
        }
        baseline_dataframe = pd.DataFrame(baseline_data)
        production_data = {
            "feature": ["col2", "col3", "", "col1"],
            "metric_name": ["FeatureImportance", "FeatureImportance", "RowCount", "FeatureImportance"],
            "metric_value": [.2, 3.7, 3, .1],
        }
        production_dataframe = pd.DataFrame(production_data)
        drift = calculate_attribution_drift(baseline_dataframe, production_dataframe)
        assert drift == 0.5135443210660507
