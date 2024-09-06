# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift evaluate metrics threshold component."""

from shared_utilities.event_utils import post_warning_event, post_email_event, add_tags_to_run, _get_run_id
from shared_utilities.constants import (
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
    PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
    TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME,
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
    SIGNAL_METRICS_GROUP,
)
from pyspark.sql import DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import (
    StringType,
    DoubleType
)


# For the list of metrics, the users expect the value should be greater than the threshold,
# We should raise alert when the metric value is less than the threshold.
# For the metrics not in the list, we should alert when the metrics value is greater than the threshold.
Metric_Value_Should_Greater_Than_Threshold = [TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME,
                                              PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
                                              NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME,
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
                                              ]


def _generate_error_message(df, signal_name: str):
    """Generate the error message for the given thresholds."""
    columns = df.schema.names
    selected_cols = [SIGNAL_METRICS_METRIC_VALUE, SIGNAL_METRICS_METRIC_NAME, SIGNAL_METRICS_THRESHOLD_VALUE]
    if SIGNAL_METRICS_GROUP in columns:
        selected_cols.append(SIGNAL_METRICS_GROUP)
    df = df.select(*selected_cols)
    features_violating_threshold = df.toJSON().collect()
    error_message = f"The signal '{signal_name}' has failed due to one or more features violating metric thresholds.\n"
    error_message += "The feature names and their corresponding computed metric values violating "
    error_message += f"the threshold are \n{features_violating_threshold}\n"

    return error_message


def _clean_metrics_df(metrics_df: DataFrame) -> DataFrame:
    """Apply some minor data cleaning to metrics DF."""
    # filter out null values and empty string
    is_not_nan_metrics_threshold_df = metrics_df.filter(
        F.col(SIGNAL_METRICS_THRESHOLD_VALUE).isNotNull()
    )

    is_not_nan_metrics_threshold_df = is_not_nan_metrics_threshold_df.filter(
        F.col(SIGNAL_METRICS_METRIC_VALUE).isNotNull()
    )

    if is_not_nan_metrics_threshold_df.schema[SIGNAL_METRICS_THRESHOLD_VALUE].dataType == StringType():
        is_not_nan_metrics_threshold_df = is_not_nan_metrics_threshold_df.filter(
            F.col(SIGNAL_METRICS_THRESHOLD_VALUE) != F.lit("")
        )

    if is_not_nan_metrics_threshold_df.schema[SIGNAL_METRICS_METRIC_VALUE].dataType == StringType():
        is_not_nan_metrics_threshold_df = is_not_nan_metrics_threshold_df.filter(
            F.col(SIGNAL_METRICS_METRIC_VALUE) != F.lit("")
        )

    return is_not_nan_metrics_threshold_df


def evaluate_metrics_threshold(
    signal_name: str, metrics_to_evaluate_df: DataFrame, notification_emails: str
):
    """Evaluate the computed metrics against the threshold."""
    print("Computed metrics to evaluate against threshold DF:")
    metrics_to_evaluate_df.show()

    cleaned_metrics_df = _clean_metrics_df(metrics_to_evaluate_df)

    metrics_threshold_breached_df = calculate_metrics_breach(cleaned_metrics_df)
    print("Metrics calculated to breach threshold DF:")
    metrics_threshold_breached_df.show()

    return send_email_for_breached(metrics_threshold_breached_df, signal_name, notification_emails)


def calculate_metrics_breach(metrics_threshold_df: DataFrame):
    """Calculate the breached metrics by the given thresholds."""
    metrics_threshold_df = metrics_threshold_df.withColumn(SIGNAL_METRICS_THRESHOLD_VALUE,
                                                           F.col(SIGNAL_METRICS_THRESHOLD_VALUE).cast(DoubleType()))
    metrics_threshold_df = metrics_threshold_df.withColumn(SIGNAL_METRICS_METRIC_VALUE,
                                                           F.col(SIGNAL_METRICS_METRIC_VALUE).cast(DoubleType()))

    metrics_threshold_breached_df = metrics_threshold_df.where(
        (F.col(SIGNAL_METRICS_METRIC_NAME).isin(Metric_Value_Should_Greater_Than_Threshold) &
         (F.col(SIGNAL_METRICS_METRIC_VALUE) < F.col(SIGNAL_METRICS_THRESHOLD_VALUE))) |
        (~F.col(SIGNAL_METRICS_METRIC_NAME).isin(Metric_Value_Should_Greater_Than_Threshold) &
         (F.col(SIGNAL_METRICS_METRIC_VALUE) > F.col(SIGNAL_METRICS_THRESHOLD_VALUE)))
        )
    return metrics_threshold_breached_df


def send_email_for_breached(metrics_threshold_breached_df: DataFrame,
                            signal_name: str,
                            notification_emails: str):
    """Send email notification for the breached metrics."""
    if metrics_threshold_breached_df.count() > 0:
        error_message = _generate_error_message(metrics_threshold_breached_df, signal_name)
        post_warning_event(error_message)
        if notification_emails is not None and notification_emails != "":
            post_email_event(signal_name, notification_emails, error_message)
        add_tags_to_run(_get_run_id(),
                        {
                            "azureml_modelmonitor_threshold_breached": error_message
                        })
        return False
    return True
