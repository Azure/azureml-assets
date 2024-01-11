# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""


from pyspark.sql import SparkSession
from src.model_monitor_evaluate_metrics_threshold.evaluate_metrics_threshold import (
    calculate_metrics_breach,
    _generate_error_message)
from src.shared_utilities.constants import (
    AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME,
    AGGREGATED_GROUNDEDNESS_PASS_RATE_METRIC_NAME,
    AGGREGATED_FLUENCY_PASS_RATE_METRIC_NAME,
    AGGREGATED_SIMILARITY_PASS_RATE_METRIC_NAME,
    AGGREGATED_RELEVANCE_PASS_RATE_METRIC_NAME,
    NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME,
    NORMALIZED_WASSERSTEN_DISTANCE_METRIC_NAME,
    JENSEN_SHANNON_DISTANCE_METRIC_NAME,
    PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
    POPULATION_STABILITY_INDEX_METRIC_NAME,
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
    TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME,
)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pytest


metrics_breached = [
            (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, 0.367, 0.5),
            (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, 0.367, 0.5),
            (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, 0.367, 0.5),
            (AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME, 3.0, 5.0),
            (AGGREGATED_GROUNDEDNESS_PASS_RATE_METRIC_NAME, 3.0, 5.0),
            (AGGREGATED_FLUENCY_PASS_RATE_METRIC_NAME, 3.0, 5.0),
            (AGGREGATED_SIMILARITY_PASS_RATE_METRIC_NAME, 3.0, 5.0),
            (AGGREGATED_RELEVANCE_PASS_RATE_METRIC_NAME, 3.0, 5.0),
            (JENSEN_SHANNON_DISTANCE_METRIC_NAME, 0.8, 0.5),
            (POPULATION_STABILITY_INDEX_METRIC_NAME, 0.8, 0.5),
            (NORMALIZED_WASSERSTEN_DISTANCE_METRIC_NAME, 0.8, 0.5),
        ]
columns = [SIGNAL_METRICS_METRIC_NAME, SIGNAL_METRICS_METRIC_VALUE, SIGNAL_METRICS_THRESHOLD_VALUE]
metrics_breached_df = create_pyspark_dataframe(metrics_breached, columns)

metrics_not_breached = [
            (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, 0.8, 0.5),
            (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, 0.8, 0.5),
            (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, 0.8, 0.5),
            (AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME, 5.0, 3.0),
            (AGGREGATED_GROUNDEDNESS_PASS_RATE_METRIC_NAME, 5.0, 3.0),
            (AGGREGATED_FLUENCY_PASS_RATE_METRIC_NAME, 5.0, 3.0),
            (AGGREGATED_SIMILARITY_PASS_RATE_METRIC_NAME, 5.0, 3.0),
            (AGGREGATED_RELEVANCE_PASS_RATE_METRIC_NAME, 5.0, 3.0),
            (JENSEN_SHANNON_DISTANCE_METRIC_NAME, 0.4, 0.5),
            (POPULATION_STABILITY_INDEX_METRIC_NAME, 0.4, 0.5),
            (NORMALIZED_WASSERSTEN_DISTANCE_METRIC_NAME, 0.4, 0.5),
        ]
columns = [SIGNAL_METRICS_METRIC_NAME, SIGNAL_METRICS_METRIC_VALUE, SIGNAL_METRICS_THRESHOLD_VALUE]
metrics_not_breached_df = create_pyspark_dataframe(metrics_not_breached, columns)
# create empty dataframe
spark = SparkSession.builder.getOrCreate()
emptyRDD = spark.sparkContext.emptyRDD()


@pytest.mark.unit1
class TestEvaluateMetricsThreshold:
    """Test class for evaluate metrics threshold component."""

    @pytest.mark.parametrize("metrics_df, breached_metrics_df",
                             [(metrics_breached_df, metrics_breached_df),
                              (metrics_not_breached_df, emptyRDD)])
    def test_calculate_metrics_breach(
            self,
            metrics_df,
            breached_metrics_df
    ):
        """Test evaluate metrics breach."""
        actual_breached_metrics_df = calculate_metrics_breach(metrics_df)
        assert breached_metrics_df.count() == actual_breached_metrics_df.count()
        assert sorted(breached_metrics_df.collect()) == sorted(actual_breached_metrics_df.collect())

    @pytest.mark.parametrize("data, schema, expected_message",
                             [([("1", "metics", 4.6, 5.6)],
                               ["group", SIGNAL_METRICS_METRIC_NAME,
                                SIGNAL_METRICS_METRIC_VALUE, SIGNAL_METRICS_THRESHOLD_VALUE],
                               'The signal \'name\' has failed due to one or more features violating ' +
                               'metric thresholds.\nThe feature names and their corresponding computed ' +
                               'metric values violating the threshold are \n[\'{"metric_value":4.6,' +
                               '"metric_name":"metics","threshold_value":5.6,"group":"1"}\']\n'),
                              ([("metics", 4.6, 5.6)],
                               [SIGNAL_METRICS_METRIC_NAME,
                                SIGNAL_METRICS_METRIC_VALUE, SIGNAL_METRICS_THRESHOLD_VALUE],
                               'The signal \'name\' has failed due to one or more features violating ' +
                               'metric thresholds.\nThe feature names and their corresponding computed ' +
                               'metric values violating the threshold are ' +
                               '\n[\'{"metric_value":4.6,"metric_name":"metics","threshold_value":5.6}\']\n')])
    def test_generate_error_message(
        self,
        data,
        schema,
        expected_message
    ):
        """Test generate error message."""
        df = create_pyspark_dataframe(data, schema)
        message = _generate_error_message(df, "name")
        assert expected_message == message
