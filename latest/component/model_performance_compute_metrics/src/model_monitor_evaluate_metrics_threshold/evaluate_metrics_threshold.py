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
    NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME,
    PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
    TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME,
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
    ACCURACY_METRIC_NAME,
    PERCISION_METRIC_NAME,
    RECALL_METRIC_NAME,
)
import pyspark
import pyspark.sql.functions as F


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
                                              ACCURACY_METRIC_NAME,
                                              PERCISION_METRIC_NAME,
                                              RECALL_METRIC_NAME
                                              ]


def _generate_error_message(df, signal_name: str):
    """Generate the error message for the given thresholds."""
    df = df.select("group", "metric_value", "metric_name", "threshold_value")
    features_violating_threshold = df.toJSON().collect()
    error_message = f"The signal '{signal_name}' has failed due to one or more features violating metric thresholds.\n"
    error_message += "The feature names and their corresponding computed metric values violating "
    error_message += f"the threshold are \n{features_violating_threshold}\n"

    return error_message


def evaluate_metrics_threshold(
    signal_name: str, metrics_to_evaluate_df, notification_emails: str
):
    """Evaluate the computed metrics against the threshold."""
    metrics_to_evaluate_df.show()

    is_nan_metrics_threshold_df = metrics_to_evaluate_df.filter(
        metrics_to_evaluate_df.threshold_value.isNull()
    )

    is_nan_metrics_threshold_df = is_nan_metrics_threshold_df.filter(
        is_nan_metrics_threshold_df.metric_value.isNull()
    )

    is_not_nan_metrics_threshold_df = metrics_to_evaluate_df.filter(
        metrics_to_evaluate_df.threshold_value.isNotNull()
    )

    is_not_nan_metrics_threshold_df = is_not_nan_metrics_threshold_df.filter(
        is_not_nan_metrics_threshold_df.metric_value.isNotNull()
    )

    metrics_threshold_breached_df = calculate_metrics_breach(is_not_nan_metrics_threshold_df)
    metrics_threshold_breached_df.show()

    return send_email_for_breached(metrics_threshold_breached_df, notification_emails, signal_name)


def calculate_metrics_breach(metrics_threshold_df: pyspark.sql.DataFrame):
    """Calculate the breached metrics by the given thresholds."""
    metrics_threshold_breached_df = metrics_threshold_df.where(
        (F.col(SIGNAL_METRICS_METRIC_NAME).isin(Metric_Value_Should_Greater_Than_Threshold) &
         (F.col(SIGNAL_METRICS_METRIC_VALUE) < F.col(SIGNAL_METRICS_THRESHOLD_VALUE))) |
        (~F.col(SIGNAL_METRICS_METRIC_NAME).isin(Metric_Value_Should_Greater_Than_Threshold) &
         (F.col(SIGNAL_METRICS_METRIC_VALUE) > F.col(SIGNAL_METRICS_THRESHOLD_VALUE)))
        )
    return metrics_threshold_breached_df


def send_email_for_breached(metrics_threshold_breached_df: pyspark.sql.DataFrame,
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
