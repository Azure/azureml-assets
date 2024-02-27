# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer identify problem traffic."""

import argparse
import requests
import json
import yaml
from pyspark.sql.functions import col, lit, udf, when, concat, explode
from typing import List
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType
)
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.gsq import apply_annotation

from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable
)
from shared_utilities.llm_utils import (
    API_KEY,
    AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
    AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    _APITokenManager,
    _WorkspaceConnectionTokenManager,
    _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    _request_api,
    get_openai_request_args
)

# Todo: remove later
VIOLATED_METRICS = ["fluency", "coherence"]

N_SAMPLES = 100

TOPIC_TEMPLATE = "\n\n".join(
    [
        "System:",
        "You are an AI assistant. You will be given a set of questions. Please categorize these quesitons into a few topics based on their intent, "  # noqa: E501
        "please try your best to avoid categorize question into its own topic whenever appropriate, and format your answer in this format:",  # noqa: E501
        '{ "<topic_0>": ["<question_00>", "<question_01>", ...], "<topic_1>": ["<question_10>", "<question_11>", ...], ... }',  # noqa: E501
        "Please only return the json content without prefix or suffix. If there are too many topics, please sort the topics with the querestion counts belong to it, in descent order, and returns the top 10."  # noqa: E501
        "User:",
        "Here are the queries:",
        "{queries}",
    ]
)


def get_output_schema() -> StructType:
    """Get Output Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("span_id", StringType(), True),
            StructField("root_question", StringType(), True),
            StructField("question", StringType(), True),
            StructField("answer", StringType(), True),
            StructField("topic_list", StringType(), True),
            StructField("group_list", StringType(), True),
            StructField("violated_metrics", StringType(), True),
            StructField("index_content", StringType(), True),
            StructField("index_id", StringType(), True),
            StructField("context", StringType(), True),
        ]
    )
    return schema


def _query_topic(
    queries,
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    model: str,
    temperature: float,
    top_p: float,
    num_samples: int,
    frequency_penalty: float,
    presence_penalty:
    float,
    max_tokens=3000,
    stop: str = None
) -> List[int]:

    # Copy request_data to avoid modifying the original dict.
    prompt = TOPIC_TEMPLATE.replace("{queries}", json.dumps(queries))

    print("prompt:", prompt)
    request_data = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "n": num_samples,
        "max_tokens": max_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    if stop:
        request_data["stop"] = stop

    response = {}
    try:
        response, time_taken = _request_api(
            session=session,
            endpoint_url=endpoint_url,
            token_manager=token_manager,
            **request_data,
        )

        # Append time taken to the line
        response["response_time_sec"] = time_taken
        print(response["samples"][0])
        topics = response["samples"][0]
    except Exception as e:  # noqa: B902
        response["finish_reason"] = ["error"]
        response["error"] = [str(e)]
        raise e

    return topics


def get_topic(questions,
              workspace_connection_arm_id,
              model_deployment_name,
              api_call_retry_max_count,
              api_call_retry_backoff_factor,
              request_args):
    
    token_manager = _WorkspaceConnectionTokenManager(
        connection_name=workspace_connection_arm_id,
        auth_header=API_KEY)
    azure_endpoint_domain_name = token_manager.get_endpoint_domain().replace("https://", "")
    azure_openai_api_version = token_manager.get_api_version()

    azure_endpoint_url = _check_and_format_azure_endpoint_url(
        AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
        AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
        azure_endpoint_domain_name,
        azure_openai_api_version,
        model_deployment_name  # mdoel
    )
    # endpoint_url = azure_endpoint_url
    httpClient = _HTTPClientWithRetry(
        n_retry=api_call_retry_max_count,
        backoff_factor=api_call_retry_backoff_factor,
    )
    """Get semantic groups for queries."""
    request_args = json.loads(request_args)
    result = ""
    with httpClient.client as session:
        result = _query_topic(
            questions,
            session, azure_endpoint_url, token_manager,
            **request_args,
        )

    return json.loads(result)


def _append_value(string_input, value):
    if string_input == "":
        return value
    else:
        string_set = set(string_input.split(","))
        string_set.add(value)
        return ",".join(string_set)


@udf(returnType=ArrayType(StringType()))
def assign_topic_and_group(topic_list, group_list, question, violated_metrics, metrics, topic_group_dict):
    """Assign topic name and group name for bad queries."""
    topic_group = json.loads(topic_group_dict)
    for group_name, (topic, q_list) in topic_group.items():
        if question in q_list and (metrics in violated_metrics):
            topic_list = _append_value(topic_list, topic)
            group_list = _append_value(group_list, group_name)
    return (topic_list, group_list)


@udf(returnType=StringType())
def assign_good_topic(topic_list, question, metrics_score, topics_dict):
    """Assign topic name for good queries."""
    topic_question = json.loads(topics_dict)
    for topic, q_list in topic_question.items():
        if question in q_list and metrics_score == 5:
            topic_list = _append_value(topic_list, topic)
    return topic_list


# Todo: temp usage, need to remove later
@udf(returnType=StructType([
    StructField("question", StringType()),
    StructField("answer", StringType()),
    StructField("text", StringType())]))
def Get_gsq_input(input, output, root_span):
    """Gsq Adapter."""
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        text_builder = ""
        for span in tree:
            # Todo: get retrieval span
            if span.span_row["name"] == "lookup":
                lookup_input = json.loads(span.span_row["input"])
                queries = lookup_input["queries"]
                lookup_outputs = json.loads(span.span_row["output"])
                if isinstance(queries, list):
                    top_k_list = lookup_outputs[0]
                    for lookup_output in top_k_list:
                        text_builder = text_builder + lookup_output["text"]
                elif isinstance(queries, str):
                    for lookup_output in lookup_outputs:
                        text_builder = text_builder + lookup_output["text"]
                break
        return json.loads(input)["question"], json.loads(output)["answer"], text_builder
    except KeyError as e:
        print("Required field not found: ", e)
        return None


@udf(returnType=ArrayType(StructType([
    StructField("span_id", StringType()),
    StructField("index_content", StringType()),
    StructField("index_id", StringType()),
    StructField("query", StringType()),
    StructField("text", StringType())])))
def parse_debugging_info(root_span):
    """Parse the span tree to get debugging info."""
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        spans_array = []
        for span in tree:
            if span.span_row["name"] == "lookup":
                span_id = span.span_row["span_id"]
                lookup_input = json.loads(span.span_row["input"])
                index_content = lookup_input["mlindex_content"]
                index_payload = yaml.safe_load(index_content)
                index_id = index_payload['index']['index']
                queries = lookup_input["queries"]
                lookup_outputs = json.loads(span.span_row["output"])
                if isinstance(queries, list):
                    for i in range(len(queries)):
                        query = queries[i]
                        top_k_list = lookup_outputs[i]
                        text_builder = ""
                        for lookup_output in top_k_list:
                            text_builder = text_builder + lookup_output["text"]
                        spans_array.append((span_id, index_content, index_id, query, text_builder))
                elif isinstance(queries, str):
                    text_builder = ""
                    for lookup_output in lookup_outputs:
                        text_builder = text_builder + lookup_output["text"]
                    spans_array.append((span_id, index_content, index_id, queries, text_builder))
        return spans_array
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def convert_to_span_level(df):
    """Convert the dataframe from trace level to span level."""
    debugging_details = parse_debugging_info(col("root_span"))
    if debugging_details is None:
        return df
    df = df.withColumn("debugging_info", debugging_details)
    df_exploaded = df.withColumn("debugging_details", explode("debugging_info")).drop("debugging_info")
    span_level_df = df_exploaded.withColumn("span_id", col("debugging_details.span_id"))\
        .withColumn("index_id", col("debugging_details.index_id"))\
        .withColumn("index_content", col("debugging_details.index_content"))\
        .withColumn("query", col("debugging_details.query"))\
        .withColumn("text", col("debugging_details.text"))\
        .drop("debugging_details")
    return span_level_df


@udf(returnType=StringType())
def assign_violated_metrics(violated_metrics, metric_score, metrics):
    """Add violated metrics name."""
    if (metric_score < 4):
        violated_metrics = _append_value(violated_metrics, metrics)
    return violated_metrics


def run():
    """Identify problem traffic."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_groups", type=str)
    parser.add_argument("--signal_scored_data", type=str)
    parser.add_argument("--signal_output", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    args = parser.parse_args()

    request_args = get_openai_request_args(args)

    # Todo: parse signal_output and get the violated metrics. Or return empty

    signal_scored_data_df = try_read_mltable_in_spark(
        args.signal_scored_data, "signal_scored_data"
    )

    signal_scored_data_df.show()
    gsq_input = Get_gsq_input(col("input"), col("output"), col("root_span"))
    signal_scored_data_df = signal_scored_data_df.withColumn("question", gsq_input["question"]) \
                                                 .withColumn("answer", gsq_input["answer"]) \
                                                 .drop("user_id") \
                                                 .drop("session_id") \
                                                 .drop("start_time") \
                                                 .drop("end_time") \
                                                 .drop("input") \
                                                 .drop("output")
    print("gsq input production df")
    signal_scored_data_df.show()

    annotations_df = apply_annotation(
        metric_names="AcceptableCoherenceScorePerInstance,AggregatedCoherencePassRate,AcceptableFluencyScorePerInstance,AggregatedFluencyPassRate",  # noqa: E501
        production_df=signal_scored_data_df,
        model_deployment_name=args.model_deployment_name,
        workspace_connection_arm_id=args.workspace_connection_arm_id,
        num_samples=args.num_samples,
        sample_rate=1.0,
        request_args=request_args,
        prompt_column_name="question",
        completion_column_name="answer",
        context_column_name="context",
        ground_truth_column_name="ground_truth"
    )
    annotations_df = annotations_df.where((col("Fluency") != -1) & (col("Coherence") != -1))
    signal_scored_output_df = annotations_df.join(signal_scored_data_df, ['trace_id'], "inner")
    # Todo: add all violated metrics
    signal_scored_output_df = signal_scored_output_df.select([col("trace_id"),
                                                              col("question"),
                                                              col("answer"),
                                                              col("Fluency").alias("fluency_score"),
                                                              col("Coherence").alias("coherence_score"),
                                                              col("root_span")])
    print("gsq_output")
    signal_scored_output_df.show()
    # signal_scored_output_df = try_read_mltable_in_spark(args.signal_scored_output, "signal_scored_output")
    df = signal_scored_output_df.withColumn("topic_list", lit("")) \
                                .withColumn("group_list", lit("")) \
                                .withColumn("violated_metrics", lit(""))

    # seperate bad groups with semantic topic
    # violated_metrics = violated_metrics_df.select(collect_list("metrics")).collect()[0][0]
    for metrics in VIOLATED_METRICS:
        print("======Current metrics=====")
        print(metrics)
        score_name = metrics+"_score"
        df = df.withColumn("violated_metrics",
                           assign_violated_metrics(col("violated_metrics"), col(score_name), lit(metrics)))

        # add good group and bad default group
        good_group_name = f"{metrics}_good_group"
        default_bad_group_name = f"{metrics}_bad_group_default"
        df = df.withColumn("group_list",
                           when((col(score_name) == 5) & (col("group_list") == ""), good_group_name)
                          .when((col(score_name) == 5) & (col("group_list") != ""), concat(col("group_list"), lit(","), lit(good_group_name)))  # noqa
                          .when((col(score_name) < 4) & (col("group_list") == ""), default_bad_group_name)  # noqa
                          .when((col(score_name) < 4) & (col("group_list") != ""), concat(col("group_list"), lit(","), lit(default_bad_group_name)))  # noqa
                          .otherwise(col("group_list")))  # noqa

        pdf = df.toPandas()
        bad_answers = pdf[pdf[score_name] < 4]
        bad_samples = bad_answers.sample(n=min(N_SAMPLES, len(bad_answers)))
        good_answers = pdf[pdf[score_name] == 5]
        # sample good samples to have same size as bad samples
        # good_samples = good_answers.sample(n=min(N_SAMPLES, len(bad_answers)))
        good_samples = good_answers

        # add semantic groups for bad queries
        topics_dict = get_topic(bad_samples["question"].tolist(),
                                args.workspace_connection_arm_id,
                                args.model_deployment_name,
                                args.api_call_retry_max_count,
                                args.api_call_retry_backoff_factor,
                                json.dumps(request_args))

        topic_group_dict = {f"{metrics}_bad_group_{i}": (k, v) for i, (k, v) in enumerate(topics_dict.items())}
        topic_group_columns = assign_topic_and_group(col("topic_list"),
                                                     col("group_list"),
                                                     col("question"),
                                                     col("violated_metrics"),
                                                     lit(metrics),
                                                     lit(json.dumps(topic_group_dict)))

        df = df.withColumn("topic_list", topic_group_columns[0])
        df = df.withColumn("group_list", topic_group_columns[1])

        # add semantic groups for good queries
        topics_dict = get_topic(good_samples["question"].tolist(),
                                args.workspace_connection_arm_id, args.model_deployment_name,
                                args.api_call_retry_max_count,
                                args.api_call_retry_backoff_factor,
                                json.dumps(request_args))

        df = df.withColumn("topic_list", assign_good_topic(col("topic_list"),
                                                           col("question"),
                                                           col(score_name),
                                                           lit(json.dumps(topics_dict))))

    sampled_df = df.filter(col("topic_list") != "")

    span_level_df = convert_to_span_level(sampled_df)
    save_spark_df_as_mltable(span_level_df, args.data_with_groups)


if __name__ == "__main__":
    run()
