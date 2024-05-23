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
    save_empty_dataframe,
    init_momo_component_environment,
)
from shared_utilities.constants import (
    P_VALUE_THRESHOLD,
    TEXT_SPLITTER,
    PROMPT_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    INDEX_ID_COLUMN,
    BAD_GROUP_COLUMN,
    GOOD_GROUP_COLUMN,
    CONFIDENCE_SCORE_COLUMN,
    GROUP_LIST_COLUMN,
    DEFAULT_RETRIEVAL_SCORE
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
            StructField(INDEX_ID_COLUMN, StringType(), True),
            StructField(BAD_GROUP_COLUMN, StringType(), True),
            StructField(GOOD_GROUP_COLUMN, StringType(), True),
            StructField(CONFIDENCE_SCORE_COLUMN, FloatType(), True)
        ]
    )
    return schema


def save_action_data(action_rows, output_path):
    """Save Action Data Spark DataFrame."""
    schema = get_output_schema()
    df = create_spark_df(action_rows, schema)
    save_spark_df_as_mltable(df, output_path)


def generate_action_rows(pdf, index_set, group_set):
    """Generate the action data."""
    actions_row = []
    for index in index_set:
        for group in group_set:
            print("Group: ", group)
            metrics = group.split("_")[0]
            topic = group.split("_")[-1]
            good_group_name = f"{metrics}_good_group"
            if group == good_group_name:
                print(f"Skip {good_group_name}, only do t-test for bad group")
                continue
            index_df = pdf[pdf[INDEX_ID_COLUMN] == index]
            good_answer_scores = index_df[index_df[GROUP_LIST_COLUMN].apply(lambda x: good_group_name in x)][INDEX_SCORE_LLM_COLUMN]  # noqa: E501
            bad_answer_scores = index_df[index_df[GROUP_LIST_COLUMN].apply(lambda x: group in x)][INDEX_SCORE_LLM_COLUMN]  # noqa: E501
            good_answer_names = index_df[index_df[GROUP_LIST_COLUMN].apply(lambda x: good_group_name in x)][[PROMPT_COLUMN, INDEX_SCORE_LLM_COLUMN]]  # noqa: E501
            bad_answer_names = index_df[index_df[GROUP_LIST_COLUMN].apply(lambda x: group in x)][[PROMPT_COLUMN, INDEX_SCORE_LLM_COLUMN]]  # noqa: E501
            print("good answer questions: ")
            print(good_answer_names)
            print("bad answer questions: ")
            print(bad_answer_names)
            t_stat, p_value = perform_ttest(good_answer_scores, bad_answer_scores)
            bad_mean = statistics.mean(bad_answer_scores)
            print("Mean value of bad group: ", bad_mean)
            if t_stat > 0 and p_value < P_VALUE_THRESHOLD and bad_mean < 3.0:
                print("Generating action for topic: ", topic)
                # entry: [index_id, bad_group, good_group, confidence_score]
                entry = [
                    index,
                    group,
                    good_group_name,
                    float(1.0 - p_value)
                ]
                actions_row.append(entry)
    return actions_row


def perform_ttest(good_answer_scores, bad_answer_scores):
    """Perform Mann-Whitney U test."""
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


def get_unique_group_and_index(data_with_action_metric_score_df):
    """Get the group set and index set."""
    group_set = set()
    index_set = set()
    for score_data_row in data_with_action_metric_score_df.collect():
        groups = score_data_row[GROUP_LIST_COLUMN].split(TEXT_SPLITTER)
        group_set.update(groups)
        index_set.add(score_data_row[INDEX_ID_COLUMN])
    return group_set, index_set


def run():
    """Correlation test."""
    # setup momo environment
    init_momo_component_environment()

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    args = parser.parse_args()

    data_with_action_metric_score_df = try_read_mltable_in_spark(
        args.data_with_action_metric_score, "data_with_action_metric_score"
    )

    if not data_with_action_metric_score_df or data_with_action_metric_score_df.isEmpty():
        print("Empty metrics score data")
        save_empty_dataframe(get_output_schema(), args.action_data)
        return

    group_set, index_set = get_unique_group_and_index(data_with_action_metric_score_df)
    print("===All groups===")
    print(group_set)
    print("===All indexes===")
    print(index_set)
    data_with_action_metric_score_df = data_with_action_metric_score_df.filter(col(INDEX_SCORE_LLM_COLUMN) > DEFAULT_RETRIEVAL_SCORE)  # noqa: E501
    pdf = data_with_action_metric_score_df.toPandas()
    print(pdf)
    action_rows = generate_action_rows(pdf, index_set, group_set)
    save_action_data(action_rows, args.action_data)


if __name__ == "__main__":
    run()
