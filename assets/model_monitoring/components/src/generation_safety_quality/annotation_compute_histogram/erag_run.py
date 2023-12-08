import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType
import json
from shared_utilities import io_utils
from LLM_helper import (
    PROMPT, COMPLETION, CONTEXT, GROUND_TRUTH, OPENAI_REQUEST_PARAMS, API_KEY,
    AZURE_OPENAI_API_COMPLETION_URL_PATTERN, AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    _WorkspaceConnectionTokenManager, _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    query_relevance_score
)


@F.udf(StringType())
def get_output(output):
    return json.loads(output)["output"]


@F.udf(DoubleType())
def get_lookup_score(node_run_info_json_str):
    node_run_info = json.loads(node_run_info_json_str)
    lookup_outputs = node_run_info["lookup"]["output"]
    total = 0.0
    count = 0
    for lookup_output in lookup_outputs:
        count += 1
        total += lookup_output["score"]
    return total / count


@F.udf(IntegerType())
def get_overall_score(question, answer, workspace_connection_arm_id, model_deployment_name,
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
        model_deployment_name  # model
    )
    httpClient = _HTTPClientWithRetry(
        n_retry=api_call_retry_max_count,
        backoff_factor=api_call_retry_backoff_factor,
    )

    request_args = json.loads(request_args)
    rating = 3.14
    with httpClient.client as session:
        rating = query_relevance_score(
            (question, answer),
            session, azure_endpoint_url, token_manager,
            **request_args,
        )
    # print(rating)
    return rating


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--metric_names", type=str, required=True)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--frequency_penalty", type=float, default=0.0)
    parser.add_argument("--presence_penalty", type=float, default=0.0)
    parser.add_argument("--stop", type=str, default=None)

    parser.add_argument("--groundedness_rating_threshold", type=int, default=4)
    parser.add_argument("--similarity_rating_threshold", type=int, default=4)
    parser.add_argument("--relevance_rating_threshold", type=int, default=4)
    parser.add_argument("--fluency_rating_threshold", type=int, default=4)
    parser.add_argument("--coherence_rating_threshold", type=int, default=4)

    parser.add_argument("--prompt_column_name", type=str, default=PROMPT)
    parser.add_argument("--completion_column_name", type=str, default=COMPLETION)
    parser.add_argument("--context_column_name", type=str, default=CONTEXT)
    parser.add_argument("--ground_truth_column_name", type=str, default=GROUND_TRUTH)

    parser.add_argument("--sample_rate", type=float, required=False, default=1.0)
    parser.add_argument(
        "--request_error_rate_threshold",
        type=float,
        default=0.5,
        help="Fail if the running error rate for the endpoint requests "
        "raises above this threshold.",
    )
    parser.add_argument("--api_call_retry_backoff_factor", type=int, default=4)
    parser.add_argument("--api_call_retry_max_count", type=int, default=10)
    parser.add_argument("--histogram", type=str, required=True)
    parser.add_argument("--samples_index", type=str, required=True)
    parser.add_argument("--groundedness_violations", type=str, required=True)
    parser.add_argument("--fluency_violations", type=str, required=True)
    parser.add_argument("--relevance_violations", type=str, required=True)
    parser.add_argument("--coherence_violations", type=str, required=True)
    parser.add_argument("--similarity_violations", type=str, required=True)

    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    # args = parser.parse_args()
    args = parser.parse_args(args=[
        '--production_dataset', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/521f5fe6-857f-478c-b0cf-18338b21d4d2/joined_data/',
        '--metric_names', 'AcceptableRelevanceScorePerInstance,AggregatedRelevancePassRate',
        '--model_deployment_name', 'gpt-35-turbo-v0301',
        '--workspace_connection_arm_id', '/subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourceGroups/azureml-rag-ci/providers/Microsoft.MachineLearningServices/workspaces/azureml-rag-westus2/connections/azure_open_ai',
        '--prompt_column_name', 'question',
        '--completion_column_name', 'output', '--context_column_name', 'context',
        '--ground_truth_column_name', 'ground_truth', '--sample_rate', '0.1', '--groundedness_rating_threshold', '4',
        '--relevance_rating_threshold', '4', '--similarity_rating_threshold', '4', '--fluency_rating_threshold', '4',
        '--coherence_rating_threshold', '4', '--histogram',
        'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/histogram/',
        '--samples_index', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/samples_index/',
        '--groundedness_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/groundedness_violations/',
        '--fluency_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/fluency_violations/',
        '--similarity_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/similarity_violations/',
        '--coherence_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/coherence_violations/',
        '--relevance_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/relevance_violations/'
    ])

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    request_args["model"] = args.model_deployment_name

    df = io_utils.try_read_mltable_in_spark_with_error(args.production_dataset, "joined_data")

    df = df.withColumn("answer", get_output(F.col("output")))\
           .withColumn("lookup_score", get_lookup_score(F.col("node_run_info")))\
           .drop("node_run_info", "run_info", "output")

    df = df.withColumn(
        "overall_score",
        get_overall_score(
            F.col("question"), F.col("answer"),
            F.lit(args.workspace_connection_arm_id), F.lit(args.model_deployment_name),
            F.lit(args.api_call_retry_max_count), F.lit(args.api_call_retry_backoff_factor),
            F.lit(json.dumps(request_args))
        )
    )

    # question, output, run_info, node_run_info
    df = io_utils.try_read_mltable_in_spark_with_error(args.production_dataset)
    # calculate_flow_input_output_relevance_score()
    # make two cohorts by good and bad relevance
    # get question, indexed doc relevance from node_run_info.lookup.output[0/1/2].score for the 2 cohorts
    # use t-test to check if they have statistics significant difference
    # if yes:
    # 	generate update index action
    # else:
    # 	generate general action


if __name__ == "__main__":
    run()
