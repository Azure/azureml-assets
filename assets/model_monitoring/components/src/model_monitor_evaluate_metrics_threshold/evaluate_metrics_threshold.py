# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift evaluate metrics threshold component."""

import pyspark.sql.functions as F
from shared_utilities.event_utils import post_warning_event, post_email_event, add_tags_to_run, _get_run_id

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

    metrics_threshold_breached_df = is_not_nan_metrics_threshold_df.where(
        F.col("metric_value") > F.col("threshold_value")
    )

    metrics_threshold_breached_df.show()

    if metrics_threshold_breached_df.count() > 0:
        error_message = _generate_error_message(metrics_to_evaluate_df, signal_name)
        post_warning_event(error_message)
        if notification_emails is not None and notification_emails != "":
            post_email_event(signal_name, notification_emails, error_message)
        add_tags_to_run(_get_run_id(),
                        {
                            f"azureml_modelmonitor_threshold_breached": error_message
                        })
        return False

    return True
