# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer metric calculation."""

import argparse
import requests
import json
from pyspark.sql.functions import collect_list, col, lit, udf, when
from typing import List, Tuple
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType
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
    OPENAI_REQUEST_PARAMS,
    RATING,
    PROMPT,
    COMPLETION,
    CONTEXT,
    MIN_RATING,
    MAX_RATING,
    _APITokenManager,
    _WorkspaceConnectionTokenManager,
    _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    _request_api,
    get_openai_request_args
)


Retrieval_document_RELEVANCE_TEMPLATE = "\n\n".join(
    [
        "System:",
        f"You are an AI assistant. You will be given the definition of an evaluation metric for assessing the \
            quality of an {CONTEXT} in a question-answering task. Your job is to compute an accurate evaluation \
                score using the provided evaluation metric.",
        f"Index quality measures how well the retrieved document {CONTEXT} provides relevant background information or knowledge to answer this question {PROMPT}. Consider whether \
            all and only the important aspects are contained in the {CONTEXT} when \
                evaluating index quality. Given the {PROMPT} and {COMPLETION}, score the index quality of the {COMPLETION} \
                    between {MIN_RATING} to {MAX_RATING} using the following {RATING} scale:",
        f"{RATING} 1: the document completely lacks information or knowledge of the question",
        f"{RATING} 2: the document mostly lacks information or knowledge of the question",
        f"{RATING} 3: the document is partially information or knowledge of the question",
        f"{RATING} 4: the document is mostly information or knowledge of the question",
        f"{RATING} 5: the document has perfect information or knowledge of the question",
        f"The score should be integer only, between {MIN_RATING} and {MAX_RATING}.",
        "## Example Task #0",
        json.dumps({
            CONTEXT: "Python is a popular general-purpose programming language that can be used for a wide variety of applications. It includes high-level data structures, dynamic typing, dynamic binding, and many more features that make it as useful for complex application development as it is for scripting or glue code that connects components together.",
            PROMPT: "How can I use the python tool in the langchain frame",
            COMPLETION: "Sorry, the provided context does not include information about question.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1,
        }),
        json.dumps({
            CONTEXT: "Marie Curie was a Polish-born physicist and chemist who pioneered research on radioactivity \
                and was the first woman to win a Nobel Prize.",
            PROMPT: "What field did Marie Curie excel in?",
            COMPLETION: "Marie Curie was a renowned painter who focused mainly on impressionist styles and \
                techniques.",
        }),
        "A good example response would be:",
        "## Example Task #0:",
        json.dumps({
            RATING: 1,
        }),
        "## Example Task #1",
        json.dumps({
            CONTEXT: "The Beatles were an English rock band formed in Liverpool in 1960, and they are widely \
                regarded as the most influential music band in history.",
            PROMPT: "Where were The Beatles formed?",
            COMPLETION: "The band The Beatles began their journey in London, England, and they changed the \
                history of music.",
        }),
        "A good example response would be:",
        "## Example Task #1:",
        json.dumps({
            RATING: 2,
        }),
        "## Example Task #2",
        json.dumps({
            CONTEXT: "The recent Mars rover, Perseverance, was launched in 2020 with the main goal of searching \
                for signs of ancient life on Mars. The rover also carries an experiment called MOXIE, which aims \
                    to generate oxygen from the Martian atmosphere.",
            PROMPT: "What are the main goals of Perseverance Mars rover mission?",
            COMPLETION: "The Perseverance Mars rover mission focuses on searching for signs of ancient life on Mars.",
        }),
        "A good example response would be:",
        "## Example Task #2",
        json.dumps({
            RATING: 3,
        }),
        "## Example Task #3",
        json.dumps({
            CONTEXT: "The Mediterranean diet is a commonly recommended dietary plan that emphasizes fruits, \
                vegetables, whole grains, legumes, lean proteins, and healthy fats. Studies have shown that it \
                    offers numerous health benefits, including a reduced risk of heart disease and improved \
                        cognitive health.",
            PROMPT: "What are the main components of the Mediterranean diet?",
            COMPLETION: "The Mediterranean diet primarily consists of fruits, vegetables, whole grains, and legumes.",
        }),
        "A good example response would be:",
        "## Example Task #3:",
        json.dumps({
            RATING: 4,
        }),
        "## Example Task #4",
        json.dumps({
            CONTEXT: "The Queen's Royal Castle is a well-known tourist attraction in the United Kingdom. It spans \
                over 500 acres and contains extensive gardens and parks. The castle was built in the 15th century \
                    and has been home to generations of royalty.",
            PROMPT: "What are the main attractions of the Queen's Royal Castle?",
            COMPLETION: "The main attractions of the Queen's Royal Castle are its expansive 500-acre grounds, \
                extensive gardens, parks, and the historical castle itself, which dates back to the 15th century \
                    and has housed generations of royalty.",
        }),
        "A good example response would be:",
        "## Example Task #4:",
        json.dumps({
            RATING: 5,
        }),
        "User:",
        "{input_samples}"
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
            StructField("index_score", IntegerType(), True),
        ]
    )
    return schema


def query_relevance_scores(
    turns: List[Tuple[str, str, str]],
    template:str,
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    # request_error_rate_threshold: float,
    model: str, temperature: float, top_p: float, num_samples: int,
    frequency_penalty: float, presence_penalty: float, max_tokens=3000, stop: str = None) -> List[int]:
    # if we count too many errors, we stop and raise an exception
    # error_count = 0
    # request_count = 0

    # request_count += 1
    # Copy request_data to avoid modifying the original dict.
    prompts = [template.replace("{input_samples", f"\n{json.dumps({'prompt': turn[0], 'completion': turn[1], 'context': turn[2]}, indent=4)}") for turn in turns]
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
            rating = json.loads(response["samples"][0])["rating"]
            # print("===response===")
            # print("rating=", rating, type(rating))
            ratings.append(rating)
        except Exception as e:  # noqa: B902
            response["finish_reason"] = ["error"]
            response["error"] = [str(e)]
            raise e
    return ratings


def query_relevance_score(
    turn: Tuple[str, str, str],
    template:str,
    session: requests.Session,
    endpoint_url: str,
    token_manager: _APITokenManager,
    # request_error_rate_threshold: float,
    model: str, temperature: float, top_p: float, num_samples: int,
    frequency_penalty: float, presence_penalty: float, max_tokens=3000, stop: str = None) -> int:
    turns = [turn]
    return query_relevance_scores(turns,template, session, endpoint_url, token_manager, model, temperature, top_p, num_samples,
                                  frequency_penalty, presence_penalty, max_tokens, stop)[0]


@udf(IntegerType())
def get_index_score(question, answer, context, workspace_connection_arm_id, model_deployment_name,
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
    rating = -1
    with httpClient.client as session:
        rating = query_relevance_score(
            (question, answer, context),
            Retrieval_document_RELEVANCE_TEMPLATE,
            session, azure_endpoint_url, token_manager,
            **request_args,
        )
    return rating


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
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    args = parser.parse_args()
    request_args = get_openai_request_args(args)

    data_with_groups_df = try_read_mltable_in_spark(args.data_with_groups, "data_with_groups")

    if not data_with_groups_df or data_with_groups_df.isEmpty():
        print("No input data found, creating an empty dataframe.")
        save_empty_dataframe(get_output_schema(), args.data_with_action_metric_score)
        return

    data_with_metric_score_df = data_with_groups_df.withColumn("index_score",
        get_index_score(col("query"), col("answer"), col("text"), lit(args.workspace_connection_arm_id), lit(args.model_deployment_name),
                          lit(args.api_call_retry_max_count), lit(args.api_call_retry_backoff_factor), lit(json.dumps(request_args))))

    data_with_metric_score_df = data_with_metric_score_df.withColumn("index_id", lit("1"))
    save_spark_df_as_mltable(data_with_metric_score_df, args.data_with_action_metric_score)

if __name__ == "__main__":
    run()
