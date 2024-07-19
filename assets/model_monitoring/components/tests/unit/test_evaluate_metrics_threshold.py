# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Monitor Data Quality Compute Metric component."""


from pyspark.sql import SparkSession
from src.model_monitor_evaluate_metrics_threshold.evaluate_metrics_threshold import (
    calculate_metrics_breach,
    _generate_error_message,
    _clean_metrics_df)
from src.shared_utilities.constants import (
    AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME,
    AGGREGATED_GROUNDEDNESS_PASS_RATE_METRIC_NAME,
    AGGREGATED_FLUENCY_PASS_RATE_METRIC_NAME,
    AGGREGATED_SIMILARITY_PASS_RATE_METRIC_NAME,
    AGGREGATED_RELEVANCE_PASS_RATE_METRIC_NAME,
    AVERAGE_COHERENCE_SCORE_METRIC_NAME,
    AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME,
    AVERAGE_FLUENCY_SCORE_METRIC_NAME,
    AVERAGE_RELEVANCE_SCORE_METRIC_NAME,
    AVERAGE_SIMILARITY_SCORE_METRIC_NAME,
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
            (AVERAGE_COHERENCE_SCORE_METRIC_NAME, 3.0, 5.0),
            (AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME, 3.0, 5.0),
            (AVERAGE_FLUENCY_SCORE_METRIC_NAME, 3.0, 5.0),
            (AVERAGE_SIMILARITY_SCORE_METRIC_NAME, 3.0, 5.0),
            (AVERAGE_RELEVANCE_SCORE_METRIC_NAME, 3.0, 5.0),
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
            (AVERAGE_COHERENCE_SCORE_METRIC_NAME, 5.0, 3.0),
            (AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME, 5.0, 3.0),
            (AVERAGE_FLUENCY_SCORE_METRIC_NAME, 3.0, 3.0),
            (AVERAGE_SIMILARITY_SCORE_METRIC_NAME, 5.0, 3.0),
            (AVERAGE_RELEVANCE_SCORE_METRIC_NAME, 5.0, 3.0),
        ]
metrics_not_breached_df = create_pyspark_dataframe(metrics_not_breached, columns)

metrics_str = [
            (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, "0.367", "0.5"),
            (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, "0.367", "0.5"),
            (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, "0.367", "0.5"),
            (AVERAGE_COHERENCE_SCORE_METRIC_NAME, "3.0", "20.0"),
            (AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME, "3.0", "5.0"),
            (AVERAGE_FLUENCY_SCORE_METRIC_NAME, "3.0", "100"),
            ("dropped_metric", "value", "threshold"),
            ("dropped_metric_2", "three", "five"),
        ]
metrics_str_df = create_pyspark_dataframe(metrics_str, columns)

expected_breached_str = [
        (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, "0.367", "0.5"),
        (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, "0.367", "0.5"),
        (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, "0.367", "0.5"),
        (AVERAGE_COHERENCE_SCORE_METRIC_NAME, "3.0", "20.0"),
        (AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME, "3.0", "5.0"),
        (AVERAGE_FLUENCY_SCORE_METRIC_NAME, "3.0", "100.0"),
    ]
expected_breached_str_df = create_pyspark_dataframe(expected_breached_str, columns)

null_data = [
    (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, None, None),
    (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, 0.8, None),
    (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, None, 0.5),
    (AVERAGE_FLUENCY_SCORE_METRIC_NAME, 3.0, ""),
    (AVERAGE_SIMILARITY_SCORE_METRIC_NAME, "", 5.0),
    (AVERAGE_RELEVANCE_SCORE_METRIC_NAME, "", ''),
]
test_null_df = create_pyspark_dataframe(null_data, columns)

# For this setup, the metric_value and threshold_value columns are StringType()
valid_clean_data = [
    (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, 0.8, 0.5),
    (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, 0.8, 0.9),
    (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, 0.5, 0.5),
    # The metrics value column in gsq metrics table is string type becasue there are null/empty values
    (AVERAGE_FLUENCY_SCORE_METRIC_NAME, "3.0", '3.0'),
]
test_clean_df = create_pyspark_dataframe(valid_clean_data, columns)
# create empty dataframe
spark = SparkSession.builder.getOrCreate()
emptyRDD = spark.sparkContext.emptyRDD()

clean_str_data = [
    (AVERAGE_COHERENCE_SCORE_METRIC_NAME, None, "5.0"),
    (AVERAGE_GROUNDEDNESS_SCORE_METRIC_NAME, "3.0", ""),
    (AVERAGE_FLUENCY_SCORE_METRIC_NAME, "3.0", "threshold"),
    (AVERAGE_SIMILARITY_SCORE_METRIC_NAME, "5.0", "5.0"),
    (AVERAGE_RELEVANCE_SCORE_METRIC_NAME, "3.0", "4.0"),
]
clean_str_df = create_pyspark_dataframe(clean_str_data, columns)

expected_clean_str_data = [
    (AVERAGE_FLUENCY_SCORE_METRIC_NAME, "3.0", "threshold"),
    (AVERAGE_SIMILARITY_SCORE_METRIC_NAME, "5.0", "5.0"),
    (AVERAGE_RELEVANCE_SCORE_METRIC_NAME, "3.0", "4.0"),
]
expected_clean_str_df = create_pyspark_dataframe(expected_clean_str_data, columns)

clean_double_data = [
    (TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME, 0.367, 0.5),
    (PEARSONS_CHI_SQUARED_TEST_METRIC_NAME, 0.367, 0.5),
    (NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME, 0.367, 0.5),
    (AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME, 3.0, 5.0),
]
clean_double_data_df = create_pyspark_dataframe(clean_double_data, columns)

@pytest.mark.unit
class TestEvaluateMetricsThreshold:
    """Test class for evaluate metrics threshold component."""

    @pytest.mark.parametrize("metrics_df, breached_metrics_df",
                             [(metrics_breached_df, metrics_breached_df),
                              (metrics_not_breached_df, emptyRDD),
                              (metrics_str_df, expected_breached_str_df)])
    def test_calculate_metrics_breach(
            self,
            metrics_df,
            breached_metrics_df
    ):
        """Test calculate metrics breach."""
        actual_breached_metrics_df = calculate_metrics_breach(metrics_df)
        actual_breached_metrics_df.show()
        assert breached_metrics_df.count() == actual_breached_metrics_df.count()
        assert sorted(breached_metrics_df.collect()) == sorted(actual_breached_metrics_df.collect())

    @pytest.mark.parametrize("metrics_df, expected_metrics_df",
                             [(test_null_df, emptyRDD),
                              (test_clean_df, test_clean_df),
                              (clean_str_df, expected_clean_str_df),
                              (clean_double_data_df, clean_double_data_df)])
    def test_clean_metrics_df(self, metrics_df, expected_metrics_df):
        """Test clean metrics dataframe."""
        actual_cleaned_metrics_df = _clean_metrics_df(metrics_df)
        assert expected_metrics_df.count() == actual_cleaned_metrics_df.count()
        assert sorted(expected_metrics_df.collect()) == sorted(actual_cleaned_metrics_df.collect())

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
