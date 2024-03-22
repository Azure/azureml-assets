# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer metric calculation."""

import argparse
import requests
import json
import re
import yaml
from pyspark.sql.functions import col, lit, udf, explode
from typing import Tuple
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType,
    ArrayType
)
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.constants import (
    TEXT_SPLITTER,
    PROMPT_COLUMN,
    COMPLETION_COLUMN,
    CONTEXT_COLUMN,
    TRACE_ID_COLUMN,
    INDEX_CONTENT_COLUMN,
    INDEX_SCORE_COLUMN,
    INDEX_ID_COLUMN,
    INVALID_METRICS_SCORE,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    ACTION_METRICS_COLUMN,
    PROPERTIES_COLUMN,
    TTEST_GROUP_ID_COLUMN,
    GROUP_COLUMN,
    QUERY_INTENTION_COLUMN,
    RETRIEVAL_SPAN_TYPE,
    ROOT_SPAN_COLUMN
)
from shared_utilities.prompts import RELEVANCE_TEMPLATE
from shared_utilities.io_utils import (
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
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


def get_output_schema() -> StructType:
    """Get Output Data Spark DataFrame Schema."""
    schema = StructType(
        [
            StructField(TRACE_ID_COLUMN, StringType(), True),
            StructField(TTEST_GROUP_ID_COLUMN, StringType(), True),
            StructField(GROUP_COLUMN, StringType(), True),
            StructField(QUERY_INTENTION_COLUMN, StringType(), True),
            StructField(ACTION_METRICS_COLUMN, FloatType(), True),
            StructField(PROPERTIES_COLUMN, StringType(), True)
        ]
    )
    return schema


def _post_process_results(output):
    parsed_score_response = re.findall(r'\d+', output.split("# Result")[-1].strip())
    if len(parsed_score_response) > 0:
        score = float(parsed_score_response[0].replace("'", "").strip())
    else:
        # Result of score is not found in the output string
        score = INVALID_METRICS_SCORE
        print("Not able to parse the score from the output string, setting score to nan")
    return score


def _query_relevance_score(
    turns: Tuple[str, str, str],
    template: str,
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    model: str,
    temperature: float,
    top_p: float,
    num_samples: int,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens=3000,
    stop: str = None
) -> int:

    turns_list = [turns]
    prompts = [template.replace("{{ query }}", turn[0]).replace("{{ history }}", "").replace("{{ FullBody }}", turn[2]) for turn in turns_list]  # noqa: E501

    print("prompts:", prompts)
    ratings = []
    for prompt in prompts:
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
            print("response")
            response["response_time_sec"] = time_taken
            rating = _post_process_results(response["samples"][0])
            print("===response===")
            print("rating=", rating, type(rating))
            ratings.append(rating)
        except Exception as e:  # noqa: B902
            response["finish_reason"] = ["error"]
            response["error"] = [str(e)]
            raise e
    return ratings[0]


def get_index_score(question,
                    answer,
                    text,
                    workspace_connection_arm_id,
                    model_deployment_name,
                    api_call_retry_max_count,
                    api_call_retry_backoff_factor,
                    request_args):
    """Calculate index score."""
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

    context_array = text.split(TEXT_SPLITTER)
    context_json = {}
    for i, context in enumerate(context_array):
        context_json[f"Document {i}"] = context
    # get the max index score for all contexts
    with httpClient.client as session:
        rating = _query_relevance_score(
            (question, answer, json.dumps(context_json)),
            RELEVANCE_TEMPLATE,
            session, azure_endpoint_url, token_manager,
            **request_args,
        )

    return rating


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
    StructField(INDEX_CONTENT_COLUMN, StringType()),
    StructField(INDEX_ID_COLUMN, StringType()),
    StructField(PROMPT_COLUMN, StringType()),
    StructField(CONTEXT_COLUMN, StringType()),
    StructField(INDEX_SCORE_COLUMN, FloatType()),
    StructField(RETRIEVAL_QUERY_TYPE_COLUMN, StringType()),
    StructField(RETRIEVAL_TOP_K_COLUMN, IntegerType())])))
def parse_debugging_info(root_span):
    """Parse the span tree to get debugging info."""
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        spans_array = []
        for span in tree:
            if span.span_type == RETRIEVAL_SPAN_TYPE:
                parent_id = span.parent_id
                if not parent_id:
                    print("No look up span found, skip action analyzer.")
                    return None
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
                index_input = json.loads(json.loads(index_span.attributes)["inputs"])
                index_content = index_input['mlindex_content']
                retrieval_query_type = index_input["query_type"]
                retrieval_top_k = index_input["top_k"]
                index_id = get_index_id(index_content)
                retrieval_info = json.loads(span.attributes)
                query = retrieval_info["retrieval.query"]
                retrieval_documents = json.loads(retrieval_info["retrieval.documents"])
                text = []
                score = []
                for document in retrieval_documents:
                    text.append(document["document.content"])
                    score.append(float(document["document.score"]))
                spans_array.append((index_content, index_id, query, TEXT_SPLITTER.join(text), max(score), retrieval_query_type, retrieval_top_k))  # noqa
        return spans_array
    except KeyError as e:
        print("Required field not found: ", e)
        return None


@udf(returnType=StringType())
def get_properties(debugging_details):
    properties = {
        INDEX_ID_COLUMN: debugging_details[INDEX_ID_COLUMN],
        INDEX_CONTENT_COLUMN: debugging_details[INDEX_CONTENT_COLUMN],
        PROMPT_COLUMN: debugging_details[PROMPT_COLUMN],
        CONTEXT_COLUMN: debugging_details[CONTEXT_COLUMN],
        INDEX_SCORE_COLUMN: debugging_details[INDEX_SCORE_COLUMN],
        RETRIEVAL_QUERY_TYPE_COLUMN: debugging_details[RETRIEVAL_QUERY_TYPE_COLUMN],
        RETRIEVAL_TOP_K_COLUMN: debugging_details[RETRIEVAL_TOP_K_COLUMN]
    }
    return json.dumps(properties)


@udf(returnType=StringType())
def get_unique_id(properties):
    return json.loads(properties).get(INDEX_ID_COLUMN, None)


def parse_meta_data(df):
    """Parse the meta data for calculating metrics score."""
    debugging_details = parse_debugging_info(col(ROOT_SPAN_COLUMN))
    if debugging_details is None:
        return df
    df = df.withColumn("debugging_info", debugging_details)

    df_exploded = df.withColumn("debugging_details", explode("debugging_info")).drop("debugging_info")

    df_with_properties = df_exploded.withColumn(PROPERTIES_COLUMN,
                                                get_properties(col("debugging_details"))).drop("debugging_details")
    df_with_id = df_with_properties.withColumn(TTEST_GROUP_ID_COLUMN, get_unique_id(col(PROPERTIES_COLUMN)))
    return df_with_id


@udf(FloatType())
def get_action_metrics_score(completion,
                             properties,
                             workspace_connection_arm_id,
                             model_deployment_name,
                             api_call_retry_max_count,
                             api_call_retry_backoff_factor,
                             request_args):
    """Calculate metrics score for action."""
    properties_dict = json.loads(properties)
    return get_index_score(properties_dict[PROMPT_COLUMN],
                           completion,
                           properties_dict[CONTEXT_COLUMN],
                           workspace_connection_arm_id,
                           model_deployment_name,
                           api_call_retry_max_count,
                           api_call_retry_backoff_factor,
                           json.loads(request_args))


def run():
    """Calculate metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--data_with_groups", type=str)
    parser.add_argument("--signal_scored_data", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=20)
    args = parser.parse_args()
    request_args = get_openai_request_args(args)

    data_with_groups_df = try_read_mltable_in_spark(
        args.data_with_groups, "data_with_groups"
    )

    if not data_with_groups_df or data_with_groups_df.isEmpty():
        print("No input data found, creating an empty dataframe.")
        save_empty_dataframe(get_output_schema(), args.data_with_action_metric_score)
        return

    signal_scored_data_df = try_read_mltable_in_spark("azureml://subscriptions/1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a/resourcegroups/yuachengtestrg/workspaces/ai-proj-eastus/datastores/workspaceblobstore/paths/azureml/5b3c73e9-8e67-47e2-a46e-5087b2288110/evaluation/", "signal_scored_data")

    df = data_with_groups_df.join(signal_scored_data_df, [TRACE_ID_COLUMN], "inner")

    # getting the metadata for metrics calculation
    df = parse_meta_data(df)
    print("data with meta data")
    df.show()
    # calculate the metrics score
    metrics_score = get_action_metrics_score(col(COMPLETION_COLUMN),
                                             col(PROPERTIES_COLUMN),
                                             lit(args.workspace_connection_arm_id),
                                             lit(args.model_deployment_name),
                                             lit(args.api_call_retry_max_count),
                                             lit(args.api_call_retry_backoff_factor),
                                             lit(json.dumps(request_args)))
    data_with_action_metric_score_df = df.withColumn(ACTION_METRICS_COLUMN, metrics_score)

    data_with_action_metric_score_df = data_with_action_metric_score_df.select(TRACE_ID_COLUMN,
                                                                               TTEST_GROUP_ID_COLUMN,
                                                                               GROUP_COLUMN,
                                                                               QUERY_INTENTION_COLUMN,
                                                                               ACTION_METRICS_COLUMN,
                                                                               PROPERTIES_COLUMN)
    # Output Schema:
    # +--------+--------------+-----+---------------+--------------+----------+
    # |trace_id|ttest_group_id|group|query_intention|action_metrics|properties|
    # +--------+--------------+-----+---------------+--------------+----------+
    print("data_with_action_metric_score")
    data_with_action_metric_score_df.show()
    save_spark_df_as_mltable(data_with_action_metric_score_df, args.data_with_action_metric_score)


if __name__ == "__main__":
    run()
