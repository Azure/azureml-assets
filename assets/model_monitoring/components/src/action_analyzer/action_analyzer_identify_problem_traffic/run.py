# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer identify problem traffic."""

import argparse
import json
import yaml
from pyspark.sql.functions import col, lit, udf, explode
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType,
    FloatType,
    IntegerType
)
from shared_utilities.constants import (
    GSQ_METRICS_LIST,
    METRICS_VIOLATION_THRESHOLD,
    RETRIEVAL_SPAN_TYPE,
    TEXT_SPLITTER,
    PROMPT_COLUMN,
    COMPLETION_COLUMN,
    CONTEXT_COLUMN,
    TRACE_ID_COLUMN,
    SPAN_ID_COLUMN,
    ROOT_QUESTION_COLUMN,
    TOPIC_LIST_COLUMN,
    GROUP_LIST_COLUMN,
    VIOLATED_METRICS_COLUMN,
    INDEX_CONTENT_COLUMN,
    INDEX_SCORE_COLUMN,
    INDEX_ID_COLUMN,
    ROOT_SPAN_COLUMN,
    GROUP_TOPIC_MIN_SAMPLE_SIZE,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    GOOD_METRICS_VALUE,
    DEFAULT_TOPIC_NAME,
    PROMPT_FLOW_INPUT_COLUMN
)
from shared_utilities.prompts import BERTOPIC_DEFAULT_PROMPT
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.store_url import StoreUrl

from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
    save_empty_dataframe
)
from shared_utilities.llm_utils import (
    API_KEY,
    _WorkspaceConnectionTokenManager
)

from bertopic import BERTopic
from openai import AzureOpenAI
from bertopic.representation import OpenAI


def get_output_schema() -> StructType:
    """Get Output Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField(TRACE_ID_COLUMN, StringType(), True),
            StructField(SPAN_ID_COLUMN, StringType(), True),
            StructField(ROOT_QUESTION_COLUMN, StringType(), True),
            StructField(PROMPT_COLUMN, StringType(), True),
            StructField(COMPLETION_COLUMN, StringType(), True),
            StructField(TOPIC_LIST_COLUMN, StringType(), True),
            StructField(GROUP_LIST_COLUMN, StringType(), True),
            StructField(VIOLATED_METRICS_COLUMN, StringType(), True),
            StructField(INDEX_CONTENT_COLUMN, StringType(), True),
            StructField(INDEX_ID_COLUMN, StringType(), True),
            StructField(CONTEXT_COLUMN, StringType(), True),
            StructField(INDEX_SCORE_COLUMN, FloatType(), True),
            StructField(RETRIEVAL_QUERY_TYPE_COLUMN, StringType(), True),
            StructField(RETRIEVAL_TOP_K_COLUMN, IntegerType(), True)
        ]
    )
    return schema


def bertopic_get_topic(queries,
                       workspace_connection_arm_id,
                       model_deployment_name):
    """Group queries in semantic groups using Bertopic."""
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


def _append_value(string_input, value):
    if string_input == "":
        return value
    else:
        string_set = set(string_input.split(TEXT_SPLITTER))
        string_set.add(value)
        return TEXT_SPLITTER.join(string_set)


@udf(returnType=ArrayType(StringType()))
def assign_bad_topic_and_group(topic_list,
                               group_list,
                               question,
                               violated_metrics,
                               metrics,
                               topics_dict,
                               llm_summary_enabled):
    """Assign topic name and group name for bad queries."""
    topic_question = json.loads(topics_dict)
    for i, (topic, q_list) in enumerate(topic_question.items()):
        if question in q_list and (metrics in violated_metrics):
            # do not show the topic name when disabling the conf
            topic = topic if llm_summary_enabled == "true" else f"topic_{i}"
            group_name = f"{metrics}_bad_group_{i}_{topic}"
            topic_list = _append_value(topic_list, topic)
            group_list = _append_value(group_list, group_name)
    return (topic_list, group_list)


@udf(returnType=StringType())
def assign_good_topic(topic_list, question, metrics_score, topics_dict, llm_summary_enabled):
    """Assign topic name for good queries."""
    topic_question = json.loads(topics_dict)
    for i, (topic, q_list) in enumerate(topic_question.items()):
        if question in q_list and metrics_score == GOOD_METRICS_VALUE:
            # do not show the topic name when disabling the conf
            topic = topic if llm_summary_enabled == "true" else f"topic_{i}"
            topic_list = _append_value(topic_list, topic)
    return topic_list


@udf(returnType=StringType())
def assign_default_group(group_list, query, metrics, good_queries, bad_queries):
    """Assign default group for good and bad queries."""
    good_query_list = json.loads(good_queries)
    bad_query_list = json.loads(bad_queries)
<<<<<<< HEAD
    if query in good_query_list:
        good_group_name = f"{metrics}_good_group"
        group_list = _append_value(group_list, good_group_name)
    elif query in bad_query_list:
        bad_group_name = f"{metrics}_bad_group_default_default"
        group_list = _append_value(group_list, bad_group_name)
=======
    if query in bad_query_list:
        bad_group_name = f"{metrics}_bad_group_default_default"
        group_list = _append_value(group_list, bad_group_name)
    elif query in good_query_list:
        good_group_name = f"{metrics}_good_group"
        group_list = _append_value(group_list, good_group_name)
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
    return group_list


@udf(returnType=StringType())
def assign_default_topic(topic_list, query_list, query):
    """Assign default topic if the sample size is too small."""
    queries = json.loads(query_list)
    if query in queries:
        topic_list = _append_value(topic_list, DEFAULT_TOPIC_NAME)
    return topic_list


def get_index_id(index_content):
    """Parse the index id from index yaml."""
    index_payload = yaml.safe_load(index_content)
    # if the asset id does not exist, use the index name
    if "self" in index_payload:
        index_id = index_payload["self"].get("asset_id", None)
    elif "index" in index_payload:
        index_id = index_payload["index"].get("index", None)
    else:
        index_id = None
    return index_id


@udf(returnType=ArrayType(StructType([
    StructField(SPAN_ID_COLUMN, StringType()),
    StructField(INDEX_CONTENT_COLUMN, StringType()),
    StructField(INDEX_ID_COLUMN, StringType()),
    StructField(PROMPT_COLUMN, StringType()),
    StructField(CONTEXT_COLUMN, StringType()),
    StructField(INDEX_SCORE_COLUMN, FloatType()),
    StructField(RETRIEVAL_QUERY_TYPE_COLUMN, StringType()),
    StructField(RETRIEVAL_TOP_K_COLUMN, IntegerType()),
    StructField(PROMPT_FLOW_INPUT_COLUMN, StringType())])))
def parse_debugging_info(root_span):
    """Parse the span tree to get debugging info."""
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        spans_array = []
        prompt_flow_input = tree.root_span.input
        for span in tree:
            if span.span_type == RETRIEVAL_SPAN_TYPE:
                parent_id = span.parent_id
                if not parent_id:
                    print("No look up span found, skip action analyzer.")
                    return None
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
<<<<<<< HEAD
                index_input = json.loads(json.loads(index_span.attributes)["inputs"])
=======
                index_input = json.loads(index_span.input)
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
                index_content = index_input['mlindex_content']
                retrieval_query_type = index_input["query_type"]
                retrieval_top_k = index_input["top_k"]
                index_id = get_index_id(index_content)
<<<<<<< HEAD
                retrieval_info = json.loads(span.attributes)
                query = retrieval_info["retrieval.query"]
                retrieval_documents = json.loads(retrieval_info["retrieval.documents"])
=======
                query = span.retrieval_query
                retrieval_documents = span.retrieval_documents
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
                text = []
                score = []
                for document in retrieval_documents:
                    text.append(document["document.content"])
                    score.append(float(document["document.score"]))
                spans_array.append((parent_id, index_content, index_id, query, TEXT_SPLITTER.join(text), max(score), retrieval_query_type, retrieval_top_k, prompt_flow_input))  # noqa
        return spans_array
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def convert_to_retrieval_span_df(df):
    """Convert the dataframe from trace level to span level."""
    debugging_details = parse_debugging_info(col(ROOT_SPAN_COLUMN))
    if debugging_details is None:
        return df
    df = df.withColumn("debugging_info", debugging_details)
    df_exploaded = df.withColumn("debugging_details", explode("debugging_info")).drop("debugging_info")
    span_level_df = df_exploaded.withColumn(SPAN_ID_COLUMN, col(f"debugging_details.{SPAN_ID_COLUMN}"))\
                                .withColumn(INDEX_ID_COLUMN, col(f"debugging_details.{INDEX_ID_COLUMN}"))\
                                .withColumn(INDEX_CONTENT_COLUMN, col(f"debugging_details.{INDEX_CONTENT_COLUMN}"))\
                                .withColumn(PROMPT_COLUMN, col(f"debugging_details.{PROMPT_COLUMN}"))\
                                .withColumn(CONTEXT_COLUMN, col(f"debugging_details.{CONTEXT_COLUMN}"))\
                                .withColumn(INDEX_SCORE_COLUMN, col(f"debugging_details.{INDEX_SCORE_COLUMN}"))\
                                .withColumn(RETRIEVAL_QUERY_TYPE_COLUMN,
                                            col(f"debugging_details.{RETRIEVAL_QUERY_TYPE_COLUMN}")) \
                                .withColumn(RETRIEVAL_TOP_K_COLUMN,
                                            col(f"debugging_details.{RETRIEVAL_TOP_K_COLUMN}"))\
                                .withColumn(PROMPT_FLOW_INPUT_COLUMN,
                                            col(f"debugging_details.{PROMPT_FLOW_INPUT_COLUMN}"))\
                                .drop("debugging_details")
    return span_level_df


@udf(returnType=StringType())
def assign_violated_metrics(violated_metrics, metric_score, metrics):
    """Add violated metrics name."""
    if (metric_score < METRICS_VIOLATION_THRESHOLD):
        violated_metrics = _append_value(violated_metrics, metrics)
    return violated_metrics


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
        print("Exception while getting the violated metrics. ", e)
        return []


def run():
    """Identify problem traffic."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_groups", type=str)
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

    # Skip action analyzer if no violated metrics
    violated_metrics = get_violated_metrics(args.signal_output, f"signals/{args.signal_name}")
    if violated_metrics == []:
        print("No violated metrics. No action will be generated.")
        save_empty_dataframe(get_output_schema(), args.data_with_groups)
        return

    print("Violated metrics found: ", violated_metrics)

    signal_scored_data_df = try_read_mltable_in_spark(args.signal_scored_data, "signal_scored_data")
    print("gsq output df")
    signal_scored_data_df.show()

    # Add topic_list, group_list, and violated_metrics columns in string type.
    # Different topics, groups, metrics will be appended to the string.
    # Schema:
    # +--------+-------+-------------+----------+---------+-------+------------------------+----------+----------+----------------+---------+  # noqa
    # |trace_id|span_id|root_question|completion|Coherence|Fluency|...(other metrics score)|topic_list|group_list|violated_metrics|root_span|  # noqa
    # +--------+-------+-------------+----------+---------+-------+------------------------+----------+----------+----------------+---------+  # noqa
    df = signal_scored_data_df.withColumn(TOPIC_LIST_COLUMN, lit("")) \
                              .withColumn(GROUP_LIST_COLUMN, lit("")) \
                              .withColumn(VIOLATED_METRICS_COLUMN, lit(""))
    # Rename to root question column
    df = df.withColumn(ROOT_QUESTION_COLUMN, col(PROMPT_COLUMN)).drop(PROMPT_COLUMN)

    for metrics in violated_metrics:
        print("======Current metrics=====")
        print(metrics)
        score_name = metrics
        # Add violated metrics for each query
        df = df.withColumn(VIOLATED_METRICS_COLUMN,
                           assign_violated_metrics(col(VIOLATED_METRICS_COLUMN), col(score_name), lit(metrics)))

        # Get the good and bad queries. Sample the good query in case the good query number is too large.
        pdf = df.toPandas()
        bad_answers = pdf[pdf[score_name] < METRICS_VIOLATION_THRESHOLD]
        good_answers = pdf[pdf[score_name] == GOOD_METRICS_VALUE]
        good_samples = good_answers.sample(n=min(len(bad_answers), len(good_answers)))
        bad_queries = bad_answers[ROOT_QUESTION_COLUMN].tolist()
        good_queries = good_samples[ROOT_QUESTION_COLUMN].tolist()
        print(f"Sample size for current metrics: {len(bad_queries)}")

        # Add default good group name and default bad group name for each query
        df = df.withColumn(GROUP_LIST_COLUMN, assign_default_group(col(GROUP_LIST_COLUMN),
                                                                   col(ROOT_QUESTION_COLUMN),
                                                                   lit(metrics),
                                                                   lit(json.dumps(good_queries)),
                                                                   lit(json.dumps(bad_queries))))

        # For bad queries, add semantic topic and separate in different semantic groups
        if len(bad_queries) < GROUP_TOPIC_MIN_SAMPLE_SIZE:
            # Skip grouing if the sample size is too small
            print(f"Bad sample size {len(bad_queries)} is less than {GROUP_TOPIC_MIN_SAMPLE_SIZE}. Skip grouping and set default topic.")  # noqa
            df = df.withColumn(TOPIC_LIST_COLUMN, assign_default_topic(col(TOPIC_LIST_COLUMN),
                                                                       lit(json.dumps(bad_queries)),
                                                                       col(ROOT_QUESTION_COLUMN)))
        else:
            print("Add semantic grouping for bad queries")
            topics_dict = bertopic_get_topic(bad_queries,
                                             args.workspace_connection_arm_id,
                                             args.model_deployment_name)

            topic_group_columns = assign_bad_topic_and_group(col(TOPIC_LIST_COLUMN),
                                                             col(GROUP_LIST_COLUMN),
                                                             col(ROOT_QUESTION_COLUMN),
                                                             col(VIOLATED_METRICS_COLUMN),
                                                             lit(metrics),
                                                             lit(json.dumps(topics_dict)),
                                                             lit(args.llm_summary_enabled))
            df = df.withColumn(TOPIC_LIST_COLUMN, topic_group_columns[0])
            df = df.withColumn(GROUP_LIST_COLUMN, topic_group_columns[1])

        # For good queries, only add semantic topic (no grouping)
        print("Add semantic topic for good queries")
        if len(good_queries) < GROUP_TOPIC_MIN_SAMPLE_SIZE:
            # Skip grouing if the sample size is too small
            print(f"Good sample size {len(good_queries)} is less than {GROUP_TOPIC_MIN_SAMPLE_SIZE}. Skip grouping and set default topic.")  # noqa
            df = df.withColumn(TOPIC_LIST_COLUMN, assign_default_topic(col(TOPIC_LIST_COLUMN),
                                                                       lit(json.dumps(good_queries)),
                                                                       col(ROOT_QUESTION_COLUMN)))
        else:
            topics_dict = bertopic_get_topic(good_queries,
                                             args.workspace_connection_arm_id,
                                             args.model_deployment_name)

            df = df.withColumn(TOPIC_LIST_COLUMN, assign_good_topic(col(TOPIC_LIST_COLUMN),
                                                                    col(ROOT_QUESTION_COLUMN),
                                                                    col(score_name),
                                                                    lit(json.dumps(topics_dict)),
                                                                    lit(args.llm_summary_enabled)))

    # Take the sampled df with assigned topic and group
    sampled_df = df.filter(col(TOPIC_LIST_COLUMN) != "")
    sampled_df.show()

    # convert the df from trace level to span level and extract retriveal info in below schema
    # Schema:
    # +--------+-------+-------------+----------+---------+-------+----------+----------+----------------+---------+------+-------------+-------------------------+  # noqa
    # |trace_id|span_id|root_question|completion|Coherence|Fluency|topic_list|group_list|violated_metrics|root_span|prompt|index_content|...(other retreival info)|  # noqa
    # +--------+-------+-------------+----------+---------+-------+----------+----------+----------------+---------+------+-------------+-------------------------+  # noqa
    span_level_df = convert_to_retrieval_span_df(sampled_df)
    save_spark_df_as_mltable(span_level_df, args.data_with_groups)


if __name__ == "__main__":
    run()
