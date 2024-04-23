# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer output actions."""

import argparse
from pyspark.sql.types import (
    StringType,
    BooleanType
)
from pyspark.sql import Window
from pyspark.sql.functions import collect_set, col, udf, max
from shared_utilities.io_utils import try_read_mltable_in_spark, np_encoder
import os
import json
import uuid
import datetime
import copy
from mlflow import MlflowClient
from shared_utilities.amlfs import amlfs_upload
from shared_utilities.constants import (
    INDEX_ID_COLUMN,
    INDEX_CONTENT_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    INDEX_ACTION_TYPE,
    ACTION_DESCRIPTION,
    TEXT_SPLITTER,
    MAX_SAMPLE_SIZE,
    BAD_GROUP_COLUMN,
    GOOD_GROUP_COLUMN,
    CONFIDENCE_SCORE_COLUMN,
    GROUP_LIST_COLUMN,
    TOPIC_LIST_COLUMN,
    VIOLATED_METRICS_COLUMN,
    ROOT_QUESTION_COLUMN,
    COMPLETION_COLUMN,
    ROOT_SPAN_COLUMN,
    ACTION_ID_COLUMN,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    PROMPT_FLOW_INPUT_COLUMN,
    DEFAULT_TOPIC_NAME
)


@udf(returnType=StringType())
def _generate_guid():
    return str(uuid.uuid4())


@udf(returnType=BooleanType())
def is_query_in_action_sample(group_list, action_sample_set):
    """Check if the query in the bad group of the action."""
    group_set = set(group_list.split(TEXT_SPLITTER))
    return len(group_set.intersection(action_sample_set)) > 0


def get_action_metadata(df):
    """Get the index set and violated metrics for action."""
    index_set = set()
    violated_metrics = set()
    for data_row in df.collect():
        index_set.add(data_row[INDEX_ID_COLUMN])
        violated_metrics.update(data_row[VIOLATED_METRICS_COLUMN].split(TEXT_SPLITTER))
    # sometimes a same query can be both good or bad. For good case, the violated_metrics is empty so need to remove
    violated_metrics.discard("")
    return index_set, violated_metrics


def is_index_asset(index_id):
    """Check if index id is asset id."""
    return index_id.startswith("azureml://")


def write_actions(action_bad_group_df, action_good_group_df, action_output_folder, aml_deployment_id, signal_name):
    """Write the action summary and action detail files."""
    index_set, violated_metrics = get_action_metadata(action_bad_group_df)
    local_path = str(uuid.uuid4())
    action_summary = {}
    for index_id in index_set:
        row = action_bad_group_df.filter(col(INDEX_ID_COLUMN) == index_id).collect()[0]
        action_id = row[ACTION_ID_COLUMN]
        query_intention = row["most_significant_group"].split("_")[-1]
        action = {
            "ActionId": action_id,
            "Type": INDEX_ACTION_TYPE,
            "Description": ACTION_DESCRIPTION.replace("{index_id}", index_id),
            "ConfidenceScore": row["max_confidence_score"],
            "ViolatedMetrics": ", ".join(violated_metrics),
            "QueryIntention": query_intention if query_intention != "default" else DEFAULT_TOPIC_NAME,
            "CreationTime": str(datetime.datetime.now()),
            "FilePath": os.path.join(action_output_folder, f"actions/{action_id}.json")
        }
        action_summary[action_id] = action
        action_detail = copy.deepcopy(action)
        action_detail["DeploymentId"] = aml_deployment_id
        action_detail["RunId"] = os.environ.get("AZUREML_RUN_ID", None)
        if is_index_asset(index_id):
            action_detail["IndexAssetId"] = row[INDEX_ID_COLUMN]
        else:
            action_detail["IndexName"] = row[INDEX_ID_COLUMN]
        action_detail["IndexContent"] = row[INDEX_CONTENT_COLUMN]
        action_detail["PositiveSamples"] = generate_samples(action_good_group_df, False)
        action_detail["NegativeSamples"] = generate_samples(action_bad_group_df, True)
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
        action_df = action_df.sort([INDEX_SCORE_LLM_COLUMN], ascending=False)
    sample_data = action_df.rdd.collect()
    for i in range(len(sample_data)):
        if i >= MAX_SAMPLE_SIZE and not is_negative_sample:
            break
        sample = {
            "Question": sample_data[i][ROOT_QUESTION_COLUMN],
            "Answer": sample_data[i][COMPLETION_COLUMN],
            "Topic": (sample_data[i][TOPIC_LIST_COLUMN].split(TEXT_SPLITTER))[0],
            "LookupScore": sample_data[i][INDEX_SCORE_LLM_COLUMN],
            "DebuggingInfo": sample_data[i][ROOT_SPAN_COLUMN],
            "RetrievalQueryType": sample_data[i][RETRIEVAL_QUERY_TYPE_COLUMN],
            "RetrievalTopK": sample_data[i][RETRIEVAL_TOP_K_COLUMN],
            "PromptFlowInput": sample_data[i][PROMPT_FLOW_INPUT_COLUMN]
        }
        if is_negative_sample:
            sample["ViolatedMetrics"] = sample_data[i][VIOLATED_METRICS_COLUMN].replace(TEXT_SPLITTER, ", ")
        samples.append(sample)
    return samples


def write_to_file(payload: dict, local_output_directory: str, file_name: str):
    """Save the action files to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    action_file = os.path.join(local_output_directory, f"{file_name}.json")
    with open(action_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


def run():
    """Merge and output actions."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--signal_name", type=str)
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

    # get the group with highest confidence score (except the default group)
    w = Window.partitionBy(INDEX_ID_COLUMN)
    max_conf_df = action_data_df.filter(~col(BAD_GROUP_COLUMN).endswith("_default"))\
                                .withColumn("max_confidence_score", max(CONFIDENCE_SCORE_COLUMN).over(w))\
                                .where(col(CONFIDENCE_SCORE_COLUMN) == col('max_confidence_score'))\
                                .withColumn("most_significant_group", col(BAD_GROUP_COLUMN))\
                                .select(INDEX_ID_COLUMN, "max_confidence_score", "most_significant_group")
    
    # if all the groups are default group, take the default group instead
    if max_conf_df.count() == 0:
        max_conf_df = action_data_df.withColumn("max_confidence_score", max(CONFIDENCE_SCORE_COLUMN).over(w))\
                                    .where(col(CONFIDENCE_SCORE_COLUMN) == col('max_confidence_score'))\
                                    .withColumn("most_significant_group", col(BAD_GROUP_COLUMN))\
                                    .select(INDEX_ID_COLUMN, "max_confidence_score", "most_significant_group")

    merged_action = action_data_df.groupby(INDEX_ID_COLUMN).agg(collect_set(BAD_GROUP_COLUMN).alias("action_bad_group_set"),  # noqa: E501
                                                                collect_set(GOOD_GROUP_COLUMN).alias("action_good_group_set"),  # noqa: E501
                                                                max(CONFIDENCE_SCORE_COLUMN).alias("action_confidence_score")).withColumn(ACTION_ID_COLUMN, _generate_guid())  # noqa: E501

    # join the action data with all debugging info and highest confident group
    merged_action = merged_action.join(max_conf_df, [INDEX_ID_COLUMN], "inner")
    action_with_group_df = data_with_action_metric_score_df.join(merged_action, [INDEX_ID_COLUMN], "inner")

    action_bad_group_df = action_with_group_df.filter(is_query_in_action_sample(col(GROUP_LIST_COLUMN),
                                                                                col("action_bad_group_set")))
    print("bad group")
    action_bad_group_df.show()

    action_good_group_df = action_with_group_df.filter(is_query_in_action_sample(col(GROUP_LIST_COLUMN),
                                                                                 col("action_good_group_set"))
                                                       & ~col(GROUP_LIST_COLUMN).contains("_bad_group"))
    print("good group")
    action_good_group_df.show()

    write_actions(action_bad_group_df,
                  action_good_group_df,
                  args.action_output,
                  args.aml_deployment_id,
                  args.signal_name)

    # add action analyzer tag for the root run
    root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID", None)
    print("Action generated, setting tag for pipeline run: ", root_run_id)
    client = MlflowClient()
    client.set_tag(root_run_id, "momo_action_analyzer_has_action", "true")


if __name__ == "__main__":
    run()
