# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the GSQ compute metrics subcomponent."""

import os
import sys
import pytest
import pandas as pd
from generation_safety_quality.annotation_compute_metrics.run import (
    compute_metrics,
    ALL_METRIC_NAMES,
    GROUP_COLUMN,
    METRIC_NAME_COLUMN,
    METRIC_VALUE_COLUMN,
    THRESHOLD_PARAMS)
import spark_mltable  # noqa, to enable spark.read.mltable
from shared_utilities.io_utils import init_spark


SIMILARITY = "Similarity"


@pytest.fixture(scope="module")
def gsq_preprocessor_test_setup():
    """Change working directory to root of the assets/model_monitoring_components."""
    original_work_dir = os.getcwd()
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    # change working directory to root of the assets/model_monitoring_components
    os.chdir(momo_work_dir)
    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    print("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", "NA"))
    module_path = os.path.join(os.getcwd(), "src")
    old_python_path = os.environ.get("PYTHONPATH", None)
    old_python_path = f"{old_python_path};" if old_python_path else ""
    os.environ["PYTHONPATH"] = f"{old_python_path}{module_path}"
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NA"))

    yield
    # change working directory back to original
    os.chdir(original_work_dir)
    # reset python path back to original
    os.environ["PYTHONPATH"] = old_python_path


@pytest.mark.gsq_test
@pytest.mark.unit
class TestGSQMetrics:
    """Test class for GSQ compute metrics component and utilities."""

    def test_gsq_compute_metrics(self, code_zip_test_setup, gsq_preprocessor_test_setup):
        """Test calling compute_metrics method on histogram table and validate result."""
        histogram_df = get_histogram_data()
        # convert to spark dataframe
        spark = init_spark()
        histogram_df = spark.createDataFrame(histogram_df)
        metric_names = [name for name in ALL_METRIC_NAMES if SIMILARITY not in name]
        metric_names = ",".join(metric_names)
        result = call_compute_metrics(histogram_df, metric_names)
        sort_columns = [METRIC_NAME_COLUMN, GROUP_COLUMN]
        expected_data = get_expected_data().sort_values(by=sort_columns)
        result_df = result.toPandas().sort_values(by=sort_columns)
        assert result_df.shape[0] == expected_data.shape[0]
        # validate equal irrespective of row order
        result_tuples = result_df.itertuples(index=False)
        expected_tuples = expected_data.itertuples(index=False)
        for rows in zip(result_tuples, expected_tuples):
            result_row = rows[0]
            expected_row = rows[1]
            for column in [GROUP_COLUMN, METRIC_NAME_COLUMN]:
                assert getattr(result_row, column) == getattr(expected_row, column)
            result_metric_value = getattr(result_row, METRIC_VALUE_COLUMN)
            expected_metric_value = getattr(expected_row, METRIC_VALUE_COLUMN)
            TOL = 0.1
            if result_metric_value != '':
                assert abs(float(result_metric_value) - float(expected_metric_value)) < TOL
            else:
                assert result_metric_value == expected_metric_value

    def test_pass_rate_includes_average_scores(self,
                                               code_zip_test_setup,
                                               gsq_preprocessor_test_setup):
        """Test average score included with AggregatedGroundednessPassRate."""
        histogram_df = get_histogram_data()
        spark = init_spark()
        histogram_df = spark.createDataFrame(histogram_df)
        metric_names = "AggregatedGroundednessPassRate"
        result = call_compute_metrics(histogram_df, metric_names)
        result_df = result.toPandas()
        assert "AverageGroundednessScore" in result_df[METRIC_NAME_COLUMN].values


def get_histogram_data():
    """Get histogram data for testing."""
    data = [
        [3, 2, 'AcceptableRelevanceScorePerInstance', 4],
        [5, 12, 'AcceptableRelevanceScorePerInstance', 4],
        [1, 0, 'AcceptableRelevanceScorePerInstance', 4],
        [2, 0, 'AcceptableRelevanceScorePerInstance', 4],
        [4, 0, 'AcceptableRelevanceScorePerInstance', 4],
        [5, 14, 'AcceptableFluencyScorePerInstance', 4],
        [1, 0, 'AcceptableFluencyScorePerInstance', 4],
        [2, 0, 'AcceptableFluencyScorePerInstance', 4],
        [3, 0, 'AcceptableFluencyScorePerInstance', 4],
        [4, 0, 'AcceptableFluencyScorePerInstance', 4],
        [5, 14, 'AcceptableCoherenceScorePerInstance', 4],
        [1, 0, 'AcceptableCoherenceScorePerInstance', 4],
        [2, 0, 'AcceptableCoherenceScorePerInstance', 4],
        [3, 0, 'AcceptableCoherenceScorePerInstance', 4],
        [4, 0, 'AcceptableCoherenceScorePerInstance', 4],
        [3, 1, 'AcceptableGroundednessScorePerInstance', 4],
        [5, 13, 'AcceptableGroundednessScorePerInstance', 4],
        [1, 0, 'AcceptableGroundednessScorePerInstance', 4],
        [2, 0, 'AcceptableGroundednessScorePerInstance', 4],
        [4, 0, 'AcceptableGroundednessScorePerInstance', 4],
        ['production_data', 14, 'RowCount', ''],
        ['reference_data', 14, 'RowCount', '']
    ]
    return pd.DataFrame(data, columns=['group', 'metric_value', 'metric_name', 'threshold_value'])


def get_expected_data():
    """Get expected data for testing."""
    data = [
        ['3', 2.0, 'AcceptableRelevanceScorePerInstance', float('NaN'), ''],
        ['5', 12.0, 'AcceptableRelevanceScorePerInstance', float('NaN'), ''],
        ['1', 0.0, 'AcceptableRelevanceScorePerInstance', float('NaN'), ''],
        ['2', 0.0, 'AcceptableRelevanceScorePerInstance', float('NaN'), ''],
        ['4', 0.0, 'AcceptableRelevanceScorePerInstance', float('NaN'), ''],
        ['5', 14.0, 'AcceptableFluencyScorePerInstance', float('NaN'), ''],
        ['1', 0.0, 'AcceptableFluencyScorePerInstance', float('NaN'), ''],
        ['2', 0.0, 'AcceptableFluencyScorePerInstance', float('NaN'), ''],
        ['3', 0.0, 'AcceptableFluencyScorePerInstance', float('NaN'), ''],
        ['4', 0.0, 'AcceptableFluencyScorePerInstance', float('NaN'), ''],
        ['5', 14.0, 'AcceptableCoherenceScorePerInstance', float('NaN'), ''],
        ['1', 0.0, 'AcceptableCoherenceScorePerInstance', float('NaN'), ''],
        ['2', 0.0, 'AcceptableCoherenceScorePerInstance', float('NaN'), ''],
        ['3', 0.0, 'AcceptableCoherenceScorePerInstance', float('NaN'), ''],
        ['4', 0.0, 'AcceptableCoherenceScorePerInstance', float('NaN'), ''],
        ['3', 1.0, 'AcceptableGroundednessScorePerInstance', float('NaN'), ''],
        ['5', 13.0, 'AcceptableGroundednessScorePerInstance', float('NaN'), ''],
        ['1', 0.0, 'AcceptableGroundednessScorePerInstance', float('NaN'), ''],
        ['2', 0.0, 'AcceptableGroundednessScorePerInstance', float('NaN'), ''],
        ['4', 0.0, 'AcceptableGroundednessScorePerInstance', float('NaN'), ''],
        ['production_data', 14.0, 'RowCount', float('NaN'), ''],
        ['reference_data', 14.0, 'RowCount', float('NaN'), ''],
        ['', 1.0, 'AggregatedFluencyPassRate', 0.7, ''],
        ['', '', 'AcceptableFluencyScorePerInstance', 4, ''],
        ['', 5.0, 'AverageFluencyScore', '', ''],
        ['', 0.9285714285714286, 'AggregatedGroundednessPassRate', 0.7, ''],
        ['', '', 'AcceptableGroundednessScorePerInstance', 4, ''],
        ['', 4.857142857142857, 'AverageGroundednessScore', '', ''],
        ['', 0.8571428571428571, 'AggregatedRelevancePassRate', 0.7, ''],
        ['', '', 'AcceptableRelevanceScorePerInstance', 4, ''],
        ['', 4.714285714285714, 'AverageRelevanceScore', '', ''],
        ['', 1.0, 'AggregatedCoherencePassRate', 0.7, ''],
        ['', '', 'AcceptableCoherenceScorePerInstance', 4, ''],
        ['', 5.0, 'AverageCoherenceScore', '', '']
    ]
    return pd.DataFrame(data, columns=['group', 'metric_value', 'metric_name',
                                       'threshold_value', 'group_dimension'])


def call_compute_metrics(histogram_df, metric_names):
    """Call compute_metrics method in GSQ component."""
    threshold_args = {threshold_name: 0.7 for threshold_name in THRESHOLD_PARAMS}
    return compute_metrics(histogram_df, threshold_args, metric_names)
