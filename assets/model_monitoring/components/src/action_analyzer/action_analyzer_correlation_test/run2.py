# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer correlation test."""

import argparse
from scipy import stats
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType
)
from pyspark.sql.functions import col
from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
    create_spark_df,
    save_empty_dataframe
)
from shared_utilities.constants import (
    P_VALUE_THRESHOLD,
    PROMPT_COLUMN,
    CONFIDENCE_SCORE_COLUMN,
    INVALID_METRICS_SCORE,
    VIOLATED_METRICS_COLUMN,
    TTEST_GROUP_ID_COLUMN,
    GOOD_GROUP_NAME,
    BAD_GROUP_NAME,
    ACTION_METRICS_COLUMN,
    QUERY_INTENTION_COLUMN
)
import statistics
from scipy.stats import mannwhitneyu
import numpy as np


def _count_values(arr):
    below_3 = np.sum(arr < 3)
    at_least_3 = np.sum(arr >= 3)
    return np.array([below_3, at_least_3])


def get_output_schema() -> StructType:
    """Get Action Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField(TTEST_GROUP_ID_COLUMN, StringType(), True),
            StructField(GROUP_COLUMN, StringType(), True),
            StructField(QUERY_INTENTION_COLUMN, StringType(), True),
            StructField(CONFIDENCE_SCORE_COLUMN, FloatType(), True)
        ]
    )
    return schema


def save_action_data(action_rows, output_path):
    """Save Action Data Spark DataFrame."""
    schema = get_output_schema()
    df = create_spark_df(action_rows, schema)
    save_spark_df_as_mltable(df, output_path)


def get_unique_values_by_column(df, column):
    """Get the unique set for a given column."""
    unique_values = set()
    for data_row in df.collect():
        unique_values.add(data_row[column])
    return unique_values


def generate_actions(df, ttest_groups, violated_metrics):
    """Perform correlation test to generate action data."""
    actions_data = []
    for ttest_group_id in ttest_groups:
        df_filtered = df.filter(col(TTEST_GROUP_ID_COLUMN) == ttest_group_id)
        for metrics in violated_metrics:
            print("Metrics: ", metrics)
            good_group_name = GOOD_GROUP_NAME.replace("{metrics}", metrics)
            bad_group_name = BAD_GROUP_NAME.replace("{metrics}", metrics)
            good_group_df = df_filtered.filter(col(GROUP_COLUMN) == good_group_name)
            bad_group_df = df_filtered.filter(col(GROUP_COLUMN) == bad_group_name)

            # correlation test for default bad group
            actions_data = perform_correlation_test(good_group_df, bad_group_df, ttest_group_id, bad_group_name, actions_data, "default")
            # correlation test for all bad subgroups
            query_intention_groups = get_unique_values_by_column(bad_group_df, QUERY_INTENTION_COLUMN)
            for query_intention in query_intention_groups:
                bad_subgroup_df = bad_group_df.filter(col(QUERY_INTENTION_COLUMN) == query_intention)
                actions_data = perform_correlation_test(good_group_df, bad_subgroup_df, ttest_group_id, bad_group_name, actions_data, query_intention)
    return actions_data


def perform_correlation_test(good_group_df,
                             bad_group_df,
                             ttest_group_id,
                             bad_group_name,
                             actions_data,
                             query_intention):
    """Pefrom correlation test for two groups of data."""
        print("good answer questions: ")
        print(good_group_df.select(ACTION_METRICS_COLUMN, PROMPT_COLUMN).collect())
        print("bad answer questions: ")
        print(bad_group_df.select(ACTION_METRICS_COLUMN, PROMPT_COLUMN).collect())

        good_answer_scores = good_group_df.select(ACTION_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()
        bad_answer_scores = bad_group_df.select(ACTION_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()

        t_stat, p_value = perform_ttest(good_answer_scores, bad_answer_scores)
        bad_mean = statistics.mean(bad_answer_scores)
        print("Mean value of bad group: ", bad_mean)
        if t_stat > 0 and p_value < P_VALUE_THRESHOLD and bad_mean < 3.0:
            print("Generating action for group: ", query_intention)
            # entry: [ttest_group_id, group_name, query_intention, confidence_score]
            entry = [
                ttest_group_id,
                bad_group_name,
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

    violated_metrics_df = try_read_mltable_in_spark(
        args.violated_metrics, "violated_metrics"
    )
    violated_metrics = violated_metrics_df.select(VIOLATED_METRICS_COLUMN).rdd.flatMap(lambda x: x).collect()

    # filter out invalid metrics score
    data_with_action_metric_score_df = data_with_action_metric_score_df.filter(col(ACTION_METRICS_COLUMN) > INVALID_METRICS_SCORE)  # noqa: E501

    ttest_groups = get_unique_values_by_column(data_with_action_metric_score_df, TTEST_GROUP_ID_COLUMN)
    
    print("===All ttest groups===")
    print(ttest_groups)

    # get actions
    action_rows = generate_actions(data_with_action_metric_score_df, ttest_groups, violated_metrics)

    # Output Schema:
    # +--------------+-----+---------------+----------------+
    # |ttest_group_id|group|query_intention|confidence_score|
    # +--------------+-----+---------------+----------------+
    save_action_data(action_rows, args.action_data)


if __name__ == "__main__":
    run()
