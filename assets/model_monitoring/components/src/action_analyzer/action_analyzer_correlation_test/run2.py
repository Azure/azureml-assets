# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer correlation test."""

import argparse
import json
from scipy import stats
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType
)
from pyspark.sql.functions import col, udf
from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
    create_spark_df,
    save_empty_dataframe
)
from action_analyzer.constants import (
    P_VALUE_THRESHOLD,
    PROMPT_COLUMN,
    CONFIDENCE_SCORE_COLUMN,
    INVALID_METRICS_SCORE,
    TTEST_GROUP_ID_COLUMN,
    GOOD_GROUP_NAME,
    BAD_GROUP_NAME,
    ACTION_METRICS_COLUMN,
    QUERY_INTENTION_COLUMN,
    TRACE_ID_LIST_COLUMN,
    GROUP_COLUMN,
    PROPERTIES_COLUMN,
    TRACE_ID_COLUMN,
    MEAN_THRESHOLD,
    ACTION_QUERY_INTENTION_COLUMN
)
from action_analyzer.utils import (
    get_unique_values_by_column,
    get_violated_metrics
)
import statistics
from scipy.stats import mannwhitneyu
import numpy as np


def _count_values(arr):
    """Helper function for Fisher's Exact Test."""
    below_3 = np.sum(arr < 3)
    at_least_3 = np.sum(arr >= 3)
    return np.array([below_3, at_least_3])


def get_output_schema() -> StructType:
    """Get Action Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField(TTEST_GROUP_ID_COLUMN, StringType(), True),
            StructField(TRACE_ID_LIST_COLUMN, StringType(), True),
            StructField(ACTION_QUERY_INTENTION_COLUMN, StringType(), True),
            StructField(CONFIDENCE_SCORE_COLUMN, FloatType(), True)
        ]
    )
    return schema


def save_action_data(action_rows, output_path):
    """Save Action Data Spark DataFrame."""
    schema = get_output_schema()
    df = create_spark_df(action_rows, schema)
    save_spark_df_as_mltable(df, output_path)


@udf(returnType=StringType())
def get_debugging_column(action_metrics, properties):
    """Get debugging information for t-test."""
    properties_dict = json.loads(properties)
    return f"{action_metrics} - {properties_dict[PROMPT_COLUMN]}"


def generate_actions(df, violated_metrics):
    """Perform correlation test to generate action data."""
    ttest_groups = get_unique_values_by_column(df, TTEST_GROUP_ID_COLUMN)
    print("===All ttest groups===")
    print(ttest_groups)
    actions_data = []
    for ttest_group_id in ttest_groups:
        df_filtered = df.filter(col(TTEST_GROUP_ID_COLUMN) == ttest_group_id)
        for metrics in violated_metrics:
            print("\nMetrics: ", metrics)
            good_group_name = GOOD_GROUP_NAME.replace("{metrics}", metrics)
            bad_group_name = BAD_GROUP_NAME.replace("{metrics}", metrics)
            good_group_df = df_filtered.filter(col(GROUP_COLUMN) == good_group_name)
            bad_group_df = df_filtered.filter(col(GROUP_COLUMN) == bad_group_name)
            # correlation test for default bad group
            actions_data = perform_correlation_test(good_group_df,
                                                    bad_group_df,
                                                    ttest_group_id,
                                                    actions_data,
                                                    "default")
            # correlation test for all bad subgroups
            query_intention_groups = get_unique_values_by_column(bad_group_df, QUERY_INTENTION_COLUMN)
            for query_intention in query_intention_groups:
                bad_subgroup_df = bad_group_df.filter(col(QUERY_INTENTION_COLUMN) == query_intention)
                actions_data = perform_correlation_test(good_group_df,
                                                        bad_subgroup_df,
                                                        ttest_group_id,
                                                        actions_data,
                                                        query_intention)
    return actions_data


def perform_correlation_test(good_group_df,
                             bad_group_df,
                             ttest_group_id,
                             actions_data,
                             query_intention):
    """Pefrom correlation test for two groups of data."""
    print("\nCorrelation test for topic: ", query_intention)
    # This part is for debugging t-test purpose only.
    print("good queries: ")
    good_group_df = good_group_df.withColumn("Debugging",
                                             get_debugging_column(col(ACTION_METRICS_COLUMN), col(PROPERTIES_COLUMN)))
    print("\n".join(good_group_df.select("Debugging").rdd.flatMap(lambda x: x).collect()))
    print("bad queries: ")
    bad_group_df = bad_group_df.withColumn("Debugging",
                                           get_debugging_column(col(ACTION_METRICS_COLUMN), col(PROPERTIES_COLUMN)))
    print("\n".join(bad_group_df.select("Debugging").rdd.flatMap(lambda x: x).collect()))

    good_answer_scores = good_group_df.select(ACTION_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()
    bad_answer_scores = bad_group_df.select(ACTION_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()

    t_stat, p_value = perform_ttest(good_answer_scores, bad_answer_scores)
    bad_mean = statistics.mean(bad_answer_scores)
    print("Mean value of bad group: ", bad_mean)
    if t_stat > 0 and p_value < P_VALUE_THRESHOLD and bad_mean < MEAN_THRESHOLD:
        print("Generating action for group: ", query_intention)
        # entry: [ttest_group_id, trace_list, query_intention, confidence_score]
        trace_list = ",".join(bad_group_df.select(TRACE_ID_COLUMN).rdd.flatMap(lambda x: x).collect())
        entry = [
            ttest_group_id,
            trace_list,
            query_intention,
            float(1.0 - p_value)
        ]
        actions_data.append(entry)
    return actions_data


def perform_ttest(good_answer_scores, bad_answer_scores):
    """ Perform Mann-Whitney U test."""
    t_stat, p_value = stats.ttest_ind(good_answer_scores, bad_answer_scores)
    print(f"Normal t-test T-statistic: {t_stat}, P-value: {p_value}")
    t_stat1, p_value1 = stats.ttest_ind(good_answer_scores, bad_answer_scores, equal_var=False)
    print(f"Welch's t-test T-statistic: {t_stat1}, P-value: {p_value1}")
    t_stat2, p_value2 = mannwhitneyu(good_answer_scores, bad_answer_scores, method='exact')
    print(f"Mann-Whitney U t-test T-statistic: {t_stat2}, P-value: {p_value2}")
    table = np.vstack((_count_values(np.array(good_answer_scores)), _count_values(np.array(bad_answer_scores))))
    # Perform Fisher's Exact Test
    t_stat3, p_value3 = stats.fisher_exact(table)
    print(f"Fisher's Exact Tes odds_ratio: {t_stat3}, P-value: {p_value3}")
    # Use Mann-Whitney U test for correlation test
    return t_stat2, p_value2


def run():
    """Correlation test."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--violated_metrics", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    args = parser.parse_args()

    data_with_action_metric_score_df = try_read_mltable_in_spark(
        args.data_with_action_metric_score, "data_with_action_metric_score"
    )
    if not data_with_action_metric_score_df or data_with_action_metric_score_df.isEmpty():
        print("Empty metrics score data")
        save_empty_dataframe(get_output_schema(), args.action_data)
        return

    violated_metrics = get_violated_metrics(args.violated_metrics)

    # filter out invalid metrics score
    data_with_action_metric_score_df = data_with_action_metric_score_df.filter(col(ACTION_METRICS_COLUMN) > INVALID_METRICS_SCORE)  # noqa: E501

    # generate actions
    action_rows = generate_actions(data_with_action_metric_score_df, violated_metrics)

    # Output Schema:
    # +--------------+----------+---------------+----------------+
    # |ttest_group_id|trace_list|query_intention|confidence_score|
    # +--------------+----------+---------------+----------------+
    save_action_data(action_rows, args.action_data)


if __name__ == "__main__":
    run()
