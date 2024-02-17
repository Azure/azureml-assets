# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer output actions."""

import argparse
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    BooleanType
)
from pyspark.sql.functions import collect_set, col, lit, udf, when, concat, mean
from shared_utilities.io_utils import try_read_mltable_in_spark, save_spark_df_as_mltable, init_spark, np_encoder
import uuid
import datetime
import copy

INDEX_ACTION_TYPE = "Index Action"
DESCRIPTION = "Poor answers are caused by poor indexing, please update the doc index with id "
MAX_SAMPLE_SIZE = 20

@udf(returnType=BooleanType())
def is_action_bad_group(group_list, action_group_set):
    group_set = set(group_list.split(","))
    return len(group_set.intersection(action_group_set)) > 0


@udf(returnType=StringType())
def generate_guid():
    return str(uuid.uuid4())


def get_unique_index(df):
    index_set = set()
    for data_row in df.collect():
        index_set.add(data_row["index_id"])
    return index_set


def write_actions(action_bad_group_df, action_good_group_df, action_output_folder, model_deployment_name, signal_name):
    index_set = get_unique_index(action_bad_group_df)
    action_summary = {}
    for index_id in index_set:
        row = action_bad_group_df.filter(col("index_id") == index_id).collect()[0]
        action_id = row["action_id"]
        confidence_score = row["action_confidence_score"]
        action = {
            "ActionId": action_id,
            "Type": INDEX_ACTION_TYPE,
            "Description": DESCRIPTION + index_id,
            "ConfidenceScore": confidence_score,
            "Signal": signal_name,
            "CreationTime": str(datetime.datetime.now()),
            "RelativePath": os.path.join(action_output_folder, f"{action_id}.json")
        }
        action_summary[action_id] = action
        action_detail = copy.deepcopy(action)
        action_detail["DeploymentId"] = model_deployment_name
        action_detail["RunId"] = os.environ.get("AZUREML_RUN_ID", None)
        action_detail["Index"] = row["index_content"]
        action_detail["PositiveSamples"] = generate_samples(action_bad_group_df, False)
        action_detail["NegativeSamples"] = generate_samples(action_good_group_df, True)
        write_to_file(action_detail, action_output_folder, action_id)
    write_to_file(action_summary, action_output_folder, "action_summary")


def generate_samples(action_df, is_negative_sample):
    samples = []
    sample_data = action_df.rdd.collect()
    for i in range(len(sample_data)):
        if i >= MAX_SAMPLE_SIZE:
            break
        sample = {
            "Question": sample_data[i]["question"],
            "Answer": sample_data[i]["answer"],
            "Topic": sample_data[i]["topic_list"],
            "DebuggingInfo": sample_data[i]["root_span"]
        }
        if is_negative_sample:
            sample["ViolatedMetrics"] = sample_data[i]["violated_metrics"]
        samples.append(sample)
    return samples


def write_to_file(payload: dict, local_output_directory: str, signal_name: str):
    """Save the signal to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    signal_file = os.path.join(local_output_directory, f"{signal_name}.json")
    with open(signal_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


def run():
    """Merge and output actions."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--signal_name", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    args = parser.parse_args()

    action_data_df = try_read_mltable_in_spark(
        args.action_data, "action_data"
    )

    if not action_data_df or action_data_df.isEmpty():
        print("Empty action data, create an empty folder.")
        os.makedirs(args.action_output, exist_ok=True)
        return

    data_with_action_metric_score_df = try_read_mltable_in_spark(
        args.data_with_action_metric_score, "data_with_action_metric_score"
    )

    # todo remove the groupby logic by using pure python or pandas
    merged_action = action_data_df.groupby("index_id").agg(collect_set("group").alias("action_group_set"),
                mean("confidence_score").alias("action_confidence_score")).withColumn("action_id", generate_guid())
    action_with_group_df = data_with_action_metric_score_df.join(merged_action, ['index_id'], "inner")

    action_bad_group_df = action_with_group_df.filter(is_action_bad_group(col("group_list"), col("action_group_set"))== True)
    action_bad_group_df.show()
    action_good_group_df = action_with_group_df.filter((col("groundedness_score") == 5)
                                                        & (col("relevance_score") == 5)
                                                        & (col("coherence_score") == 5)
                                                        & (col("fluency_score") == 5)
                                                        & (col("similarity_score") == 5))
    action_good_group_df.show()

    write_actions(action_bad_group_df, action_good_group_df, args.action_output, args.model_deployment_name, args.signal_name)


if __name__ == "__main__":
    run()
