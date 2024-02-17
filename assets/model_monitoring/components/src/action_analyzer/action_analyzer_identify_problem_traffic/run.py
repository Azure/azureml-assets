# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer identify problem traffic."""

import argparse
import requests
import json
import yaml
from pyspark.sql.functions import collect_list, col, lit, udf, when, concat, explode
from typing import List, Tuple
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    ArrayType
)
from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark,
    create_spark_df,
    save_empty_dataframe
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

N_SAMPLES = 100

TOPIC_TEMPLATE = "\n\n".join(
    [
        "System:",
        "You are an AI assistant. You will be given a set of questions. Please categorize these quesitons into a few topics based on their intent, "
        "please try your best to avoid categorize question into its own topic whenever appropriate, and format your answer in this json format:",
        '{ "<topic_0>": ["<question_00>", "<question_01>", ...], "<topic_1>": ["<question_10>", "<question_11>", ...], ... }',
        "If there are too many topics, please sort the topics with the querestion counts belong to it, in descent order, and returns the top 10."
        "User:",
        f"Here are the queries:", 
        "{queries}",
    ]
)

def get_output_schema() -> StructType:
    """Get Action Data Spark DataFrame Schema."""
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


def query_topic(
    queries,
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    # request_error_rate_threshold: float,
    model: str, temperature: float, top_p: float, num_samples: int,
    frequency_penalty: float, presence_penalty: float, max_tokens=3000, stop: str = None) -> List[int]:

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
            {"role": "user", "content": prompt }
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
        topics = response["samples"][0] #json.loads(response["samples"][0])
    except Exception as e:  # noqa: B902
        response["finish_reason"] = ["error"]
        response["error"] = [str(e)]
        raise e

    return topics


def get_topic(questions, workspace_connection_arm_id, model_deployment_name,
                      api_call_retry_max_count, api_call_retry_backoff_factor, request_args):
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

    request_args = json.loads(request_args)
    result = ""
    with httpClient.client as session:
        result = query_topic(
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
    topic_group = json.loads(topic_group_dict)
    for group_name, (topic, q_list) in topic_group.items():
        if question in q_list and (metrics in violated_metrics):
            topic_list = _append_value(topic_list, topic)
            group_list = _append_value(group_list, group_name)
    return (topic_list, group_list)


@udf(returnType=StringType())
def assign_good_topic(topic_list, question, metrics_score, topics_dict):
    topic_question = json.loads(topics_dict)
    for topic, q_list in topic_question.items():
        if question in q_list and metrics_score == 5:
            topic_list = _append_value(topic_list, topic)
    return topic_list


@udf(returnType=StructType([
    StructField("span_id", StringType()),
    StructField("index_content", StringType()),
    StructField("index_id", StringType()),
    StructField("query_text_pairs", ArrayType(StructType([StructField("query", StringType()),
                                              StructField("text", StringType())])))]))
def parse_debugging_info(debugging_info):
    try:
        # Todo: utilize the span tree iterator to find all the "vector_lookup" span
        vector_lookup = json.loads(debugging_info)
        index_content = vector_lookup["input"]["mlindex_content"]
        index_payload = yaml.safe_load(index_content)
        index_id = index_payload['self']['asset_id']
        queries = vector_lookup["input"]["queries"]
        lookup_outputs = vector_lookup["output"]
        query_text_output = []
        if isinstance(queries, list):
            for i in range(len(queries)):
                query = queries[i]
                top_k_list = lookup_outputs[i]
                text_builder = ""
                for lookup_output in top_k_list:
                    text_builder = text_builder + lookup_output["text"]
                query_text_output.append((query, text_builder))
        elif isinstance(queries, str):
            text_builder = ""
            for lookup_output in lookup_outputs:
                    text_builder = text_builder + lookup_output["text"]
            query_text_output.append((queries, text_builder))
        return "span_id", index_content, index_id, query_text_output
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def convert_to_span_level(df):
    debugging_details = parse_debugging_info(col("root_span"))
    df = df.withColumn("span_id", debugging_details["span_id"])\
        .withColumn("index_id", debugging_details["index_id"])\
        .withColumn("index_content", debugging_details["index_content"])\
        .withColumn("query_text_pairs", debugging_details["query_text_pairs"])
    df_exploaded = df.withColumn("query_text", explode("query_text_pairs")).drop("query_text_pairs")
    span_level_df = df_exploaded.withColumn("query", col("query_text.query")).withColumn("text", col("query_text.text")).drop("query_text")
    return span_level_df


@udf(returnType=StringType())
def assign_violated_metrics(violated_metrics, metric_score, metrics):
    if (metric_score < 4):
        violated_metrics = _append_value(violated_metrics, metrics)
    return violated_metrics


def run():
    """Identify problem traffic."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_groups", type=str)
    parser.add_argument("--production_data", type=str)
    parser.add_argument("--signal_scored_output", type=str)
    parser.add_argument("--violated_metrics_names", type=str)
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

    violated_metrics_df = try_read_mltable_in_spark(args.violated_metrics_names, "violated_metrics_names")
    if not violated_metrics_df or violated_metrics_df.isEmpty():
        print("No violated metrics, creating an empty action dataframe.")
        save_empty_dataframe(get_output_schema(), args.data_with_groups)
        return

    # only production data may have input data user error
    production_data_df = try_read_mltable_in_spark_with_error(args.production_data, "production_data")

    signal_scored_output_df = try_read_mltable_in_spark(args.signal_scored_output, "signal_scored_output")
    df = signal_scored_output_df.withColumn("topic_list", lit("")).withColumn("group_list", lit("")).withColumn("violated_metrics", lit(""))

    trace_log = try_read_mltable_in_spark("azureml://subscriptions/1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a/resourcegroups/yuachengtestrg/workspaces/momo-eastus/datastores/workspaceblobstore/paths/LocalUpload/591d756286cd48c38e4b790ab5742018/mltable_aggregated_trace_log/", "trace")
    root_span = trace_log.rdd.collect()[0]["root_span"]
    df = df.withColumn("root_span", lit(root_span))

    print("Show df")
    df.show()

    # seperate bad groups with semantic topic
    violated_metrics = violated_metrics_df.select(collect_list("metrics")).collect()[0][0]
    for metrics in violated_metrics:
        score_name = metrics+"_score"
        df = df.withColumn("violated_metrics", assign_violated_metrics(col("violated_metrics"), col(score_name), lit(metrics)))

        # add good group and bad default group
        good_group_name = f"{metrics}_good_group"
        default_bad_group_name = f"{metrics}_bad_group_default"
        df = df.withColumn("group_list",
            when((col(score_name) == 5) & (col("group_list") == ""), good_group_name)
            .otherwise(when((col(score_name) == 5) & (col("group_list") != ""), concat(col("group_list"), lit(","), lit(good_group_name)))
            .otherwise(when((col(score_name) < 4) & (col("group_list") == ""), default_bad_group_name)
            .otherwise(when((col(score_name) < 4) & (col("group_list") != ""), concat(col("group_list"), lit(","), lit(default_bad_group_name)))
            .otherwise(col("group_list"))))))

        pdf = df.toPandas()
        bad_answers = pdf[pdf[score_name] < 4]
        bad_samples = bad_answers.sample(n=min(N_SAMPLES, len(bad_answers)))
        good_answers = pdf[pdf[score_name] == 5]
        # sample good samples to have same size as bad samples
        #good_samples = good_answers.sample(n=min(N_SAMPLES, len(bad_answers)))
        good_samples = good_answers

        # add semantic groups for bad queries
        topics_dict = get_topic(bad_samples["question"].tolist(), args.workspace_connection_arm_id, args.model_deployment_name,
                    args.api_call_retry_max_count, args.api_call_retry_backoff_factor, json.dumps(request_args))
        topic_group_dict = {f"{metrics}_bad_group_{i}": (k, v) for i, (k, v) in enumerate(topics_dict.items())}
        topic_group_columns = assign_topic_and_group(col("topic_list"), col("group_list"), col("question"), col("violated_metrics"), lit(metrics), lit(json.dumps(topic_group_dict)))
        df = df.withColumn("topic_list", topic_group_columns[0])
        df = df.withColumn("group_list", topic_group_columns[1])

        # add semantic groups for good queries
        topics_dict = get_topic(good_samples["question"].tolist(), args.workspace_connection_arm_id, args.model_deployment_name,
                        args.api_call_retry_max_count, args.api_call_retry_backoff_factor, json.dumps(request_args))
        df = df.withColumn("topic_list", assign_good_topic(col("topic_list"), col("question"), col(score_name), lit(json.dumps(topics_dict))))

    sampled_df = df.filter(col("topic_list") != "")
    # azure_text = "Azure is Microsoft's cloud platform, offering a comprehensive suite of over 200 products and services designed to address today's challenges and shape the future."
    # sampled_df = sampled_df.withColumn("text", lit(azure_text))

    span_level_df = convert_to_span_level(sampled_df)
    save_spark_df_as_mltable(span_level_df, args.data_with_groups)

if __name__ == "__main__":
    run()
