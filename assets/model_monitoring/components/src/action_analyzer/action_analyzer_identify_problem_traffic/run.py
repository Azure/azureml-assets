# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer identify problem traffic."""

import argparse
import json
from pyspark.sql.functions import col, lit, udf, rand
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)
from action_analyzer.constants import (
    GSQ_METRICS_LIST,
    METRICS_VIOLATION_THRESHOLD,
    PROMPT_COLUMN,
    TRACE_ID_COLUMN,
    VIOLATED_METRICS_COLUMN,
    ROOT_SPAN_COLUMN,
    GROUP_TOPIC_MIN_SAMPLE_SIZE,
    GOOD_METRICS_VALUE,
    DEFAULT_TOPIC_NAME,
    GROUP_COLUMN,
    QUERY_INTENTION_COLUMN,
    GOOD_GROUP_NAME,
    BAD_GROUP_NAME
)
from action_analyzer.prompts import BERTOPIC_DEFAULT_PROMPT
from model_data_collector_preprocessor.store_url import StoreUrl

from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
    create_spark_df
)
from shared_utilities.llm_utils import (
    API_KEY,
    _WorkspaceConnectionTokenManager
)
try:
    from bertopic import BERTopic
    from openai import AzureOpenAI
    from bertopic.representation import OpenAI
except Exception:
    print("Error while importing Bertopic")


def get_output_schema() -> StructType:
    """Get Output Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField(TRACE_ID_COLUMN, StringType(), True),
            StructField(GROUP_COLUMN, StringType(), True),
            StructField(QUERY_INTENTION_COLUMN, StringType(), True),
        ]
    )
    return schema


def save_violated_metrics(violated_metrics, output_path):
    """Save violated metrics into spark dataframe."""
    schema = StructType(
        [
            StructField(VIOLATED_METRICS_COLUMN, StringType(), True)
        ]
    )
    data = []
    for metrics in violated_metrics:
        data.append([metrics])
    df = create_spark_df(data, schema)
    save_spark_df_as_mltable(df, output_path)


def bertopic_get_topic(queries,
                       workspace_connection_arm_id,
                       model_deployment_name):
    """Group queries in semantic groups using Bertopic."""
    try:
        token_manager = _WorkspaceConnectionTokenManager(connection_name=workspace_connection_arm_id,
                                                         auth_header=API_KEY)
        azure_endpoint_domain_name = token_manager.get_endpoint_domain()
        azure_openai_api_version = token_manager.get_api_version()
        azure_openai_api_key = token_manager.get_token()
        client = AzureOpenAI(api_version=azure_openai_api_version,
                             api_key=azure_openai_api_key,
                             azure_endpoint=azure_endpoint_domain_name,
                             azure_deployment=model_deployment_name)
        representation_model = OpenAI(client, model=model_deployment_name, chat=True, prompt=BERTOPIC_DEFAULT_PROMPT)
        topic_model = BERTopic(
            min_topic_size=round(0.15*len(queries)),
            top_n_words=5,
            representation_model=representation_model
        )
        topics, probs = topic_model.fit_transform(queries)

        docs = topic_model.get_document_info(queries)
        docs['Representation'] = docs['Representation'].str.get(0)
        doc_per_topic = docs.groupby('Representation')['Document'].agg(lambda x: list(x)).reset_index()
        topics_df = doc_per_topic.set_index('Representation')
        topics_dict = topics_df.to_dict()["Document"]

        print("Get topic dictionary: ")
        for k, v in topics_dict.items():
            print("Topic: ")
            print(k)
            print("\n")
            print("Questions: ", len(v))
            print("\t", "\n\t".join(v))
            print("\n")
        return topics_dict
    except Exception as e:
        print("Exception in Bertopic.", e)
        return None


@udf(returnType=StringType())
def get_query_intention(query, topics_dict, llm_summary_enabled):
    """Get semantic group for each query."""
    topic_query_dict = json.loads(topics_dict)
    for i, (topic, query_list) in enumerate(topic_query_dict.items()):
        if query in query_list:
            # do not show the topic name when disabling the conf
            return topic if llm_summary_enabled == "true" else f"topic_{i}"
    return DEFAULT_TOPIC_NAME


def get_violated_metrics(signal_out_url, signal_name):
    """Get the violated metrics names from the gsq output."""
    violated_metrics = []
    try:
        store_url = StoreUrl(signal_out_url)
        gsq_output = store_url.read_file_content(f"{signal_name}.json")
        gsq_output_json = json.loads(gsq_output)
        metrics_dict = gsq_output_json["metrics"]
        for metrics in GSQ_METRICS_LIST:
            pass_rate_metrics = f"Aggregated{metrics}PassRate"
            if pass_rate_metrics in metrics_dict:
                if metrics_dict[pass_rate_metrics]["value"] < metrics_dict[pass_rate_metrics]["threshold"]:
                    print(f"Metrics {metrics} violated.")
                    violated_metrics.append(metrics)
        return violated_metrics
    except Exception as e:
        print("Exception while getting the violated metrics.", e)
        return []


def add_query_intention(df, workspace_connection_arm_id, model_deployment_name, llm_summary_enabled):
    """Add query intention for each query."""
    if df.count() < GROUP_TOPIC_MIN_SAMPLE_SIZE:
        # Skip grouing if the sample size is too small
        print(f"Sample size {df.count()} is less than {GROUP_TOPIC_MIN_SAMPLE_SIZE}. Skip grouping and set default topic.")  # noqa
        df = df.withColumn(QUERY_INTENTION_COLUMN, lit(DEFAULT_TOPIC_NAME))
    else:
        print("Add semantic groups.")
        queries = df.select(PROMPT_COLUMN).rdd.flatMap(lambda x: x).collect()
        topics_dict = bertopic_get_topic(queries,
                                         workspace_connection_arm_id,
                                         model_deployment_name)

        if topics_dict is not None:
            df = df.withColumn(QUERY_INTENTION_COLUMN, get_query_intention(col(PROMPT_COLUMN),
                                                                           lit(json.dumps(topics_dict)),
                                                                           lit(llm_summary_enabled)))
        else:
            # when bertopic failed, add default topic name
            df = df.withColumn(QUERY_INTENTION_COLUMN, lit(DEFAULT_TOPIC_NAME))
    return df


def run():
    """Identify problem traffic."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_groups", type=str)
    parser.add_argument("--violated_metrics", type=str)
    parser.add_argument("--signal_scored_data", type=str)
    parser.add_argument("--signal_output", type=str)
    parser.add_argument("--signal_name", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    parser.add_argument("--llm_summary_enabled", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    args = parser.parse_args()

    data_with_groups_df = create_spark_df([], get_output_schema())
    # Skip action analyzer if no violated metrics
    violated_metrics = get_violated_metrics(args.signal_output, f"signals/{args.signal_name}")
    if violated_metrics == []:
        print("No violated metrics. No action will be generated.")
        save_spark_df_as_mltable(data_with_groups_df, args.data_with_groups)
        return

    print("Violated metrics found: ", violated_metrics)
    save_violated_metrics(violated_metrics, args.violated_metrics)

    signal_scored_data_df = try_read_mltable_in_spark(args.signal_scored_data, "signal_scored_data")
    print("gsq output df")
    signal_scored_data_df.show()

    # Add group and query_intention
    # Schema:
    # +--------+------+----------+---------+-------+------------------------+-----+---------------+
    # |trace_id|prompt|completion|Coherence|Fluency|...(other metrics score)|group|query_intention|
    # +--------+------+----------+---------+-------+------------------------+-----+---------------+
    df = signal_scored_data_df.withColumn(GROUP_COLUMN, lit("")) \
                              .withColumn(QUERY_INTENTION_COLUMN, lit("")) \
                              .drop(ROOT_SPAN_COLUMN)

    for metrics in violated_metrics:
        print("======Current metrics=====")
        print(metrics)
        score_name = metrics

        # Get the good and bad queries.
        # Sample the good queries as same number of bad queires in case the good query number is too large.
        df_good = df.filter(col(score_name) == GOOD_METRICS_VALUE)
        df_bad = df.filter(col(score_name) < METRICS_VIOLATION_THRESHOLD)
        df_good = df_good.orderBy(rand()).limit(df_bad.count())
        print(f"Sample size for current metrics: {df_bad.count()}")

        # assign the query group (good or bad).
        good_group = GOOD_GROUP_NAME.replace("{metrics}", metrics)
        bad_group = BAD_GROUP_NAME.replace("{metrics}", metrics)
        df_good = df_good.withColumn(GROUP_COLUMN, lit(good_group))
        df_bad = df_bad.withColumn(GROUP_COLUMN, lit(bad_group))

        # Add query intention for good and bad groups.
        df_good = add_query_intention(df_good,
                                      args.workspace_connection_arm_id,
                                      args.model_deployment_name,
                                      args.llm_summary_enabled)
        df_bad = add_query_intention(df_bad,
                                     args.workspace_connection_arm_id,
                                     args.model_deployment_name,
                                     args.llm_summary_enabled)

        print("bad df")
        df_bad.show()
        print("good df")
        df_good.show()
        # append to output dataframe
        df_good = df_good.select(TRACE_ID_COLUMN, GROUP_COLUMN, QUERY_INTENTION_COLUMN)
        df_bad = df_bad.select(TRACE_ID_COLUMN, GROUP_COLUMN, QUERY_INTENTION_COLUMN)
        data_with_groups_df = data_with_groups_df.union(df_good).union(df_bad)

    # Output Schema:
    # +--------+-----+---------------+
    # |trace_id|group|query_intention|
    # +--------+-----+---------------+
    save_spark_df_as_mltable(data_with_groups_df, args.data_with_groups)


if __name__ == "__main__":
    run()
