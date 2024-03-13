# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer metric calculation."""

import argparse
import requests
import json
import re
from pyspark.sql.functions import col, lit, udf
from typing import Tuple
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    FloatType
)
from shared_utilities.constants import (
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
    INDEX_SCORE_LLM_COLUMN,
    INDEX_ID_COLUMN,
    DEFAULT_RETRIEVAL_SCORE
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


def _post_process_results(output):
    parsed_score_response = re.findall(f"\d+", output.split("# Result")[-1].strip())  # noqa
    if len(parsed_score_response) > 0:
        score = float(parsed_score_response[0].replace("'", "").strip())
    else:
        # Result of score is not found in the output string
        score = DEFAULT_RETRIEVAL_SCORE
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


@udf(FloatType())
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

    request_args = json.loads(request_args)
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
            StructField(INDEX_SCORE_LLM_COLUMN, IntegerType(), True),
        ]
    )
    return schema


def run():
    """Calculate metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--data_with_groups", type=str)
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

    index_score = get_index_score(col(PROMPT_COLUMN),
                                  col(COMPLETION_COLUMN),
                                  col(CONTEXT_COLUMN),
                                  lit(args.workspace_connection_arm_id),
                                  lit(args.model_deployment_name),
                                  lit(args.api_call_retry_max_count),
                                  lit(args.api_call_retry_backoff_factor),
                                  lit(json.dumps(request_args)))
    data_with_metric_score_df = data_with_groups_df.withColumn(INDEX_SCORE_LLM_COLUMN, index_score)

    print("data_with_metric_score_df")
    data_with_metric_score_df.show()
    save_spark_df_as_mltable(data_with_metric_score_df, args.data_with_action_metric_score)


if __name__ == "__main__":
    run()
