import os
import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, DoubleType, IntegerType, ArrayType
import json
from datetime import datetime
import tempfile
from scipy import stats
from azureml.fsspec import AzureMachineLearningFileSystem
from shared_utilities import io_utils
from LLM_helper import (
    PROMPT, COMPLETION, CONTEXT, GROUND_TRUTH, OPENAI_REQUEST_PARAMS, API_KEY,
    AZURE_OPENAI_API_COMPLETION_URL_PATTERN, AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    _WorkspaceConnectionTokenManager, _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    query_relevance_score, query_topic
)


@F.udf(StringType())
def get_answer(output):
    try:
        return json.loads(output)["output"]
    except KeyError:
        print("no output")
        return None


@F.udf(DoubleType())
def get_lookup_score(node_run_info_json_str):
    node_run_info = json.loads(node_run_info_json_str)
    try:
        lookup_outputs = node_run_info["lookup"]["output"]
        total = 0.0
        count = 0
        for lookup_output in lookup_outputs:
            count += 1
            total += lookup_output["score"]
        return total / count
    except KeyError:
        print("no output")
        return -1.0


@F.udf(ArrayType(DoubleType()))
def get_embedding(node_run_info_json_str):
    node_run_info = json.loads(node_run_info_json_str)
    try:
        api_calls = node_run_info["lookup"]["api_calls"]
        # TODO search for the element with name "search"
        embedding = api_calls[0]["children"][0]["output"]["data"][0]["embedding"]
        return embedding
    except KeyError:
        print("no embedding")
        return None


def get_index_content(node_run_info_str):
    node_run_info = json.loads(node_run_info_str)
    return node_run_info["lookup"]["inputs"]["mlindex_content"]


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
    rating = -1
    with httpClient.client as session:
        rating = query_relevance_score(
            (question, answer),
            session, azure_endpoint_url, token_manager,
            **request_args,
        )
    # print(rating)
    return rating


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
        model_deployment_name  # model
    )
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


class Action:
    def __init__(self, title, description, samples=None, index=None, topics=None):
        self.title = title
        self.description = description
        self.index = index
        self.samples = samples
        self.topics = topics


class ActionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Action):
            return obj.__dict__
        return super().default(obj)


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
    args = parser.parse_args()
    # args = parser.parse_args(args=[
    #     '--production_dataset', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/521f5fe6-857f-478c-b0cf-18338b21d4d2/joined_data/',
    #     '--metric_names', 'AcceptableRelevanceScorePerInstance,AggregatedRelevancePassRate',
    #     '--model_deployment_name', 'gpt-35-turbo-v0301',
    #     '--workspace_connection_arm_id', '/subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourceGroups/azureml-rag-ci/providers/Microsoft.MachineLearningServices/workspaces/azureml-rag-westus2/connections/azure_open_ai',
    #     '--prompt_column_name', 'question',
    #     '--completion_column_name', 'output', '--context_column_name', 'context',
    #     '--ground_truth_column_name', 'ground_truth', '--sample_rate', '0.1', '--groundedness_rating_threshold', '4',
    #     '--relevance_rating_threshold', '4', '--similarity_rating_threshold', '4', '--fluency_rating_threshold', '4',
    #     '--coherence_rating_threshold', '4', '--histogram',
    #     'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/histogram/',
    #     '--samples_index', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/samples_index/',
    #     '--groundedness_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/groundedness_violations/',
    #     '--fluency_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/fluency_violations/',
    #     '--similarity_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/similarity_violations/',
    #     '--coherence_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/coherence_violations/',
    #     '--relevance_violations', 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/azureml/48b56276-c08b-44ed-afc8-f7e03013932b/relevance_violations/'
    # ])

    request_args = {
        arg: getattr(args, arg) for arg in OPENAI_REQUEST_PARAMS if hasattr(args, arg)
    }
    request_args["model"] = args.model_deployment_name

    df = io_utils.try_read_mltable_in_spark_with_error(args.production_dataset, "joined_data")

    node_run_info_json_str = df.first()["node_run_info"]
    index = get_index_content(node_run_info_json_str)

    df = df.withColumn("answer", get_answer(F.col("output")))\
           .withColumn("lookup_score", get_lookup_score(F.col("node_run_info")))\
           .withColumn("embedding", get_embedding(F.col("node_run_info")))\
           .withColumn("index", F.lit(index))\
           .drop("node_run_info", "run_info", "output", "chat_history", "flow_info", "flow_run_info")

    df = df.withColumn(
        "overall_score",
        get_overall_score(
            F.col("question"), F.col("answer"),
            F.lit(args.workspace_connection_arm_id), F.lit(args.model_deployment_name),
            F.lit(args.api_call_retry_max_count), F.lit(args.api_call_retry_backoff_factor),
            F.lit(json.dumps(request_args))
        )
    )

    pdf = df.toPandas()
    # Separate the data into groups
    good_answers = pdf[pdf['overall_score'] >= 4]['lookup_score']
    bad_answers = pdf[pdf['overall_score'] < 4][['lookup_score', 'question', 'answer', 'embedding', 'overall_score']]

    # get bad samples for topic categorization
    n_sampels = 100
    print(len(bad_answers))
    bad_samples = bad_answers[[
        'question', 'answer', 'embedding', 'overall_score'
    ]].sample(n=min(n_sampels, len(bad_answers)))
    bad_samples['embedding'] = bad_samples['embedding'].apply(list)

    # get the content of the index meta yaml
    index_yaml = pdf.loc[0]["index"]

    # Perform the t-test
    t_stat, p_value = stats.ttest_ind(good_answers, bad_answers['lookup_score'])

    print(f"T-statistic: {t_stat}, P-value: {p_value}")

    if t_stat > 0 and p_value < 0.05:
        action = Action(
            title="Update the doc index",
            description="Poor answers are caused by poor indexing, please update the doc index",
            samples=bad_samples.to_dict(orient='records'),
            index=index_yaml,
        )
    else:
        action = Action(
            title="Check steps other than doc index",
            description="Poor answers are not caused by poor indexing, please check other steps in the flow, "
                        "or maybe the questions are not clear enough",
            samples=bad_samples.to_dict(orient='records'),
        )

    # add topics
    topics_dict = get_topic(
        [sample["question"] for sample in action.samples],
        args.workspace_connection_arm_id,
        args.model_deployment_name,
        args.api_call_retry_max_count,
        args.api_call_retry_backoff_factor,
        json.dumps(request_args)
    )

    print("====topics dict====")
    print(json.dumps(topics_dict, indent=2))
    # sort topics by number of questions in each topic
    topics = {k: len(topics_dict[k]) for k in topics_dict.keys()}
    print(topics)
    sorted_topics = sorted(topics, key=topics.get, reverse=True)
    print(sorted_topics)
    action.topics = sorted_topics

    # dump above action to a json file in the default store
    # save action to local json file
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    # File name with timestamp
    file_name = f"action_{timestamp}.json"
    # Writing the JSON object to a temp file
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a file with the specific name in the temporary directory
        file_path = os.path.join(temp_dir, file_name)

        with open(file_path, "w") as f:
            json.dump(action, f, cls=ActionEncoder, indent=2)

        # upload local action json to store with AzureMachineLearningFileSystem
        # des_path = 'azureml://subscriptions/79a1ba0c-35bb-436b-bff2-3074d5ff1f89/resourcegroups/azureml-rag-ci/workspaces/azureml-rag-westus2/datastores/workspaceblobstore/paths/erag_actions/'
        des_path = "azureml://datastores/workspaceblobstore/paths/erag_actions/"  # azureml short path

        fs = AzureMachineLearningFileSystem(des_path)
        print("des path:", des_path)

        fs.upload(
            lpath=file_path,
            rpath="",
            **{"overwrite": "MERGE_WITH_OVERWRITE"},
            recursive=True,
        )


if __name__ == "__main__":
    run()
