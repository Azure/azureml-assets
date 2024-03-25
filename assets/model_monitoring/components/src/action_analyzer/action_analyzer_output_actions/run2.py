# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer output actions."""

import argparse
from pyspark.sql.functions import collect_set, col, first, lit
from shared_utilities.io_utils import try_read_mltable_in_spark, np_encoder, create_spark_df
import os
import json
import uuid
import datetime
import copy
from mlflow import MlflowClient
from shared_utilities.amlfs import amlfs_upload
from action_analyzer.utils import (
    get_unique_values_by_column,
    get_violated_metrics
)
from action_analyzer.constants import (
    INDEX_CONTENT_COLUMN,
    INDEX_ACTION_TYPE,
    ACTION_DESCRIPTION,
    MAX_SAMPLE_SIZE,
    CONFIDENCE_SCORE_COLUMN,
    COMPLETION_COLUMN,
    ROOT_SPAN_COLUMN,
    ACTION_ID_COLUMN,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    TTEST_GROUP_ID_COLUMN,
    GROUP_COLUMN,
    QUERY_INTENTION_COLUMN,
    TRACE_ID_LIST_COLUMN,
    DEFAULT_TOPIC_NAME,
    ACTION_QUERY_INTENTION_COLUMN,
    ACTION_ANALYZER_ACTION_TAG,
    PROPERTIES_COLUMN,
    TRACE_ID_COLUMN,
    ACTION_METRICS_COLUMN,
    ROOT_PROMPT_COLUMN
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType
)


def is_index_asset(index_id):
    """Check if index id is asset id."""
    return index_id.startswith("azureml://")


def write_actions(action_bad_group_df,
                  action_good_group_df,
                  violated_metrics,
                  action_output_folder,
                  aml_deployment_id):
    """Write the action summary and action detail files."""
    action_ids = get_unique_values_by_column(action_bad_group_df, ACTION_ID_COLUMN)
    local_path = str(uuid.uuid4())
    action_summary = {}
    for action_id in action_ids:
        # The same trace id will have multiple rows for different metrics, so do a groupby
        action_bad_df = action_bad_group_df.filter(col(ACTION_ID_COLUMN) == action_id) \
                                           .groupby(TRACE_ID_COLUMN) \
                                           .agg(collect_set(GROUP_COLUMN).alias(GROUP_COLUMN),
                                                first(TTEST_GROUP_ID_COLUMN).alias(TTEST_GROUP_ID_COLUMN),
                                                first(QUERY_INTENTION_COLUMN).alias(QUERY_INTENTION_COLUMN),
                                                first(ACTION_METRICS_COLUMN).alias(ACTION_METRICS_COLUMN),
                                                first(PROPERTIES_COLUMN).alias(PROPERTIES_COLUMN),
                                                first(ACTION_ID_COLUMN).alias(ACTION_ID_COLUMN),
                                                first(ACTION_QUERY_INTENTION_COLUMN).alias(ACTION_QUERY_INTENTION_COLUMN),  # noqa: E501
                                                first(CONFIDENCE_SCORE_COLUMN).alias(CONFIDENCE_SCORE_COLUMN))
        action_good_df = action_good_group_df.filter(col(ACTION_ID_COLUMN) == action_id) \
                                             .groupby(TRACE_ID_COLUMN) \
                                             .agg(collect_set(GROUP_COLUMN).alias(GROUP_COLUMN),
                                                  first(TTEST_GROUP_ID_COLUMN).alias(TTEST_GROUP_ID_COLUMN),
                                                  first(QUERY_INTENTION_COLUMN).alias(QUERY_INTENTION_COLUMN),
                                                  first(ACTION_METRICS_COLUMN).alias(ACTION_METRICS_COLUMN),
                                                  first(PROPERTIES_COLUMN).alias(PROPERTIES_COLUMN),
                                                  first(ACTION_ID_COLUMN).alias(ACTION_ID_COLUMN),
                                                  first(ACTION_QUERY_INTENTION_COLUMN).alias(ACTION_QUERY_INTENTION_COLUMN),  # noqa: E501
                                                  first(CONFIDENCE_SCORE_COLUMN).alias(CONFIDENCE_SCORE_COLUMN))

        row = action_bad_df.filter(col(ACTION_ID_COLUMN) == action_id).collect()[0]
        index_id = row[TTEST_GROUP_ID_COLUMN]
        action = {
            "ActionId": action_id,
            "Type": INDEX_ACTION_TYPE,
            "Description": ACTION_DESCRIPTION.replace("{index_id}", index_id),
            "ConfidenceScore": row[CONFIDENCE_SCORE_COLUMN],
            "ViolatedMetrics": ", ".join(violated_metrics),
            "QueryIntention": row[ACTION_QUERY_INTENTION_COLUMN],
            "CreationTime": str(datetime.datetime.now()),
            "FilePath": os.path.join(action_output_folder, f"actions/{action_id}.json")
        }
        action_summary[action_id] = action
        action_detail = copy.deepcopy(action)
        action_detail["DeploymentId"] = aml_deployment_id
        action_detail["RunId"] = os.environ.get("AZUREML_RUN_ID", None)
        if is_index_asset(index_id):
            action_detail["IndexAssetId"] = index_id
        else:
            action_detail["IndexName"] = index_id
        action_detail["IndexContent"] = json.loads(row[PROPERTIES_COLUMN])[INDEX_CONTENT_COLUMN]
        action_detail["PositiveSamples"] = generate_samples(action_good_df, False)
        action_detail["NegativeSamples"] = generate_samples(action_bad_df, True)
        print("Writing action detail of action: ")
        print(action)
        write_to_file(action_detail, local_path, action_id)
    print("Writing action summary to location ", action_output_folder)
    print(action_summary)
    write_to_file(action_summary, local_path, "action_summary")
    target_remote_path = os.path.join(action_output_folder, "actions")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)


def generate_samples(action_df, is_negative_sample):
    """Generate positive and negative samples in action file."""
    samples = []
    # sort the good samples by index score
    if not is_negative_sample:
        action_df = action_df.sort([ACTION_METRICS_COLUMN], ascending=False)
    sample_data = action_df.rdd.collect()
    for i in range(len(sample_data)):
        if i >= MAX_SAMPLE_SIZE and not is_negative_sample:
            break
        properties_dict = json.loads(sample_data[i][PROPERTIES_COLUMN])
        sample = {
            "Question": properties_dict[ROOT_PROMPT_COLUMN],
            "Answer": properties_dict[COMPLETION_COLUMN],
            "Topic": sample_data[i][QUERY_INTENTION_COLUMN],
            "LookupScore": sample_data[i][ACTION_METRICS_COLUMN],
            "DebuggingInfo": properties_dict[ROOT_SPAN_COLUMN],
            "RetrievalQueryType": properties_dict[RETRIEVAL_QUERY_TYPE_COLUMN],
            "RetrievalTopK": properties_dict[RETRIEVAL_TOP_K_COLUMN]
        }
        if is_negative_sample:
            # now only select one violated metrics. Todo: get the full violated list
            metrics_name = [group.replace("_bad", "") for group in sample_data[i][GROUP_COLUMN]]
            sample["ViolatedMetrics"] = ", ".join(metrics_name)
        samples.append(sample)
    return samples


def write_to_file(payload: dict, local_output_directory: str, file_name: str):
    """Save the action files to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    action_file = os.path.join(local_output_directory, f"{file_name}.json")
    with open(action_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


def merge_actions(action_df):
    """Merge actions with same t-test group id. Set the query intention using the group with most bad queries."""
    ttest_group_ids = get_unique_values_by_column(action_df, TTEST_GROUP_ID_COLUMN)
    actions = []
    for ttest_group_id in ttest_group_ids:
        df = action_df.filter(col(TTEST_GROUP_ID_COLUMN) == ttest_group_id)
        max_group_topic = DEFAULT_TOPIC_NAME
        max_group_count = 0
        trace_id_set = set()
        confidence_score = 0
        for data_row in df.collect():
            confidence_score += data_row[CONFIDENCE_SCORE_COLUMN]
            trace_id_list = data_row[TRACE_ID_LIST_COLUMN].split(",")
            trace_id_set.update(trace_id_list)
            # skip the default group, find the group with most bad queries
            if data_row[QUERY_INTENTION_COLUMN] == "default":
                continue
            if max_group_count < len(trace_id_list):
                max_group_count = len(trace_id_list)
                max_group_topic = data_row[QUERY_INTENTION_COLUMN]
        action_id = str(uuid.uuid4())
        action_confidence = float(confidence_score/df.count())
        action = [action_id, ttest_group_id, ",".join(trace_id_set), max_group_topic, action_confidence]
        actions.append(action)

    schema = StructType(
        [
            StructField(ACTION_ID_COLUMN, StringType(), True),
            StructField(TTEST_GROUP_ID_COLUMN, StringType(), True),
            StructField(TRACE_ID_LIST_COLUMN, StringType(), True),
            StructField(ACTION_QUERY_INTENTION_COLUMN, StringType(), True),
            StructField(CONFIDENCE_SCORE_COLUMN, FloatType(), True)
        ]
    )
    return create_spark_df(actions, schema)


def add_action_analyzer_tag():
    """Add a tag to AML root run when action generated by action analyzer."""
    root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID", None)
    print("Action generated, setting tag for pipeline run: ", root_run_id)
    client = MlflowClient()
    client.set_tag(root_run_id, ACTION_ANALYZER_ACTION_TAG, "true")


def run():
    """Merge and output actions."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--violated_metrics", type=str)
    parser.add_argument("--aml_deployment_id", type=str)
    args = parser.parse_args()

    action_data_df = try_read_mltable_in_spark(
        args.action_data, "action_data"
    )
    if not action_data_df or action_data_df.isEmpty():
        print("Empty action data, create an empty folder.")
        return

    data_with_action_metric_score_df = try_read_mltable_in_spark(
        args.data_with_action_metric_score, "data_with_action_metric_score"
    )
    violated_metrics = get_violated_metrics(args.violated_metrics)

    # Merge the actions.
    merged_action_df = merge_actions(action_data_df)
    # Join to get action metadata.
    action_df = data_with_action_metric_score_df.join(merged_action_df, [TTEST_GROUP_ID_COLUMN], "inner")

    # Get bad and good samples and write action files
    action_bad_group_df = action_df.filter(col(TRACE_ID_LIST_COLUMN).contains(col(TRACE_ID_COLUMN))
                                           & col(GROUP_COLUMN).contains(lit("bad")))
    print("bad sample df")
    action_bad_group_df.show()
    action_good_group_df = action_df.filter(~col(TRACE_ID_LIST_COLUMN).contains(col(TRACE_ID_COLUMN)))
    print("good sample df")
    action_good_group_df.show()

    write_actions(action_bad_group_df,
                  action_good_group_df,
                  violated_metrics,
                  args.action_output,
                  args.aml_deployment_id)

    # add action analyzer tag when action generated
    add_action_analyzer_tag()


if __name__ == "__main__":
    run()
