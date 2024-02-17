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
from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark,
    create_spark_df,
    save_empty_dataframe
)


def get_output_schema() -> StructType:
    """Get Action Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField("index_id", StringType(), True),
            StructField("group", StringType(), True),
            StructField("confidence_score", FloatType(), True)
        ]
    )
    return schema


def save_action_data(action_rows, output_path):
    """Save Action Data Spark DataFrame."""
    schema = get_output_schema()
    df = create_spark_df(action_rows, schema)
    save_spark_df_as_mltable(df, output_path)


def generate_action_rows(pdf, index_set, group_set):
    actions_row = []
    for index in index_set:
        for group in group_set:
            metrics = group.split("_")[0]
            good_group_name = f"{metrics}_good_group"
            if group == good_group_name:
                print(f"Skip {good_group_name}, only do t-test for bad group")
                continue
            index_df = pdf[pdf['index_id'] == index]
            good_answer_scores = index_df[index_df['group_list'].apply(lambda x: good_group_name in x)]['index_score']
            bad_answer_scores = index_df[index_df['group_list'].apply(lambda x: group in x)]['index_score']
            t_stat, p_value = perform_ttest(good_answer_scores, bad_answer_scores)
            if t_stat > 0 and p_value < 0.05:
                # entry: [index_id, group, confidence_score]
                entry = [
                    index,
                    group,
                    1-p_value
                ]
                actions_row.append(entry)
    return actions_row


def perform_ttest(good_answer_scores, bad_answer_scores):
    t_stat, p_value = stats.ttest_ind(good_answer_scores, good_answer_scores)
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    return t_stat, p_value


def get_unique_group_and_index(data_with_action_metric_score_df):
    group_set = set()
    index_set = set()
    for score_data_row in data_with_action_metric_score_df.collect():
        groups = score_data_row["group_list"].split(",")
        group_set.update(groups)
        index_set.add(score_data_row["index_id"])
    return group_set, index_set


def run():
    """Correlation test."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    args = parser.parse_args()

    data_with_action_metric_score_df = try_read_mltable_in_spark(
        args.data_with_action_metric_score, "data_with_action_metric_score"
    )

    if data_with_action_metric_score_df.isEmpty():
        print("Empty metrics score data")
        save_empty_dataframe(get_output_schema(), args.action_data)
        return

    group_set, index_set = get_unique_group_and_index(data_with_action_metric_score_df)
    print("===All groups===")
    print(group_set)
    print("===All indexes===")
    print(index_set)
    pdf = data_with_action_metric_score_df.toPandas()
    print(pdf)
    action_rows = generate_action_rows(pdf, index_set, group_set)
    save_action_data(action_rows, args.action_data)


if __name__ == "__main__":
    run()
