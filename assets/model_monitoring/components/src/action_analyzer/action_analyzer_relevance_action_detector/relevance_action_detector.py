# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for relevance action detector."""

import os
import re
import json
import yaml
from scipy import stats
import statistics
from scipy.stats import mannwhitneyu
import uuid
import datetime
import copy
import requests

from mlflow import MlflowClient
from shared_utilities.amlfs import amlfs_upload
from shared_utilities.constants import (
    METRICS_VIOLATION_THRESHOLD,
    RETRIEVAL_SPAN_TYPE,
    TEXT_SPLITTER,
    PROMPT_COLUMN,
    COMPLETION_COLUMN,
    SPAN_ID_COLUMN,
    INDEX_CONTENT_COLUMN,
    INDEX_SCORE_COLUMN,
    INDEX_ID_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    ROOT_SPAN_COLUMN,
    GROUP_TOPIC_MIN_SAMPLE_SIZE,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    GOOD_METRICS_VALUE,
    DEFAULT_TOPIC_NAME,
    PROMPT_FLOW_INPUT_COLUMN,
    DOCUMENT_RELEVANCE_SCORE_COLUMN,
    RETRIEVAL_DOC_COLUMN,
    MODIFIED_PROMPT_COLUMN,
    QUERY_INTENTION_COLUMN,
    API_CALL_RETRY_BACKOFF_FACTOR,
    API_CALL_RETRY_MAX_COUNT,
    ACTION_DESCRIPTION,
    MAX_SAMPLE_SIZE,
    DEFAULT_SCORE,
    P_VALUE_THRESHOLD
)
from shared_utilities.prompts import BERTOPIC_DEFAULT_PROMPT, RELEVANCE_TEMPLATE, RELEVANCE_METRIC_TEMPLATE
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.io_utils import (
    np_encoder
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
    get_llm_request_args
)
from bertopic import BERTopic
from openai import AzureOpenAI
from bertopic.representation import OpenAI
from typing import Tuple
from action_analyzer.action import Action, ActionType


class IndexAction(Action):
    """Index action class."""

    def __init__(self, index_id, query_ids, query_intention, confidence_score):
        """Create an index action."""
        self.index_id = index_id
        super().__init__(ActionType.IndexAction, query_ids, query_intention, confidence_score)


class DebuggingInfo:
    """Debugging info class."""

    def __init__(self,
                 span_id,
                 index_content,
                 index_id,
                 query,
                 retrieval_documents,
                 retrieval_score,
                 retrieval_query_type,
                 retrieval_top_k,
                 prompt_flow_input):
        """Create debugging info."""
        self.span_id = span_id
        self.index_content = index_content
        self.index_id = index_id
        self.query = query
        self.retrieval_documents = retrieval_documents
        self.retrieval_score = retrieval_score
        self.retrieval_query_type = retrieval_query_type
        self.retrieval_top_k = retrieval_top_k
        self.prompt_flow_input = prompt_flow_input


    def to_json_str(self):
        """Get a dict which represents the debugging class."""
        return json.dumps(self.__dict__)


def _query_llm_score(
    prompt: str,
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
    except Exception as e:  # noqa: B902
        response["finish_reason"] = ["error"]
        response["error"] = [str(e)]
        raise e
    return response


def _post_process_retrieval_results(response):
    output = response["samples"][0]
    parsed_score_response = re.findall(r'\d+', output.split("# Result")[-1].strip())
    if len(parsed_score_response) > 0:
        score = float(parsed_score_response[0].replace("'", "").strip())
    else:
        # Result of score is not found in the output string
        score = DEFAULT_SCORE
        print("Not able to parse the retrieval score, setting score to 0")
    return score


def _post_process_document_relevance_results(response):
    try:
        score = float(response["samples"][0])
    except Exception:
        score = DEFAULT_SCORE
        print("Not able to parse the document relevance score, setting to 0.")
    return score


def _prepare_llm_template_inputs(row):
    question = row[MODIFIED_PROMPT_COLUMN]
    answer = row[COMPLETION_COLUMN]
    text = row[RETRIEVAL_DOC_COLUMN]

    context_array = text.split(TEXT_SPLITTER)
    context_json = {}
    for i, context in enumerate(context_array):
        context_json[f"Document {i}"] = context
    return (question, answer, json.dumps(context_json))


def get_document_relevance_score(row,
                                 token_manager,
                                 azure_endpoint_url,
                                 http_client,
                                 request_args):
    """Get document relevance score using llm."""
    question, answer, context_json = _prepare_llm_template_inputs(row)
    prompt = RELEVANCE_METRIC_TEMPLATE.replace("{{question}}", question).replace("{{answer}}",answer).replace("{{context}}", context_json)  # noqa: E501

    with http_client.client as session:
        response = _query_llm_score(
            prompt,
            session,
            azure_endpoint_url,
            token_manager,
            **request_args,
        )
        score = _post_process_document_relevance_results(response)
    return score


def get_retrieval_score(row,
                        token_manager,
                        azure_endpoint_url,
                        http_client,
                        request_args):
    """Get retrieval score using llm."""
    question, answer, context_json = _prepare_llm_template_inputs(row)
    prompt = RELEVANCE_TEMPLATE.replace("{{ query }}", question).replace("{{ history }}", "").replace("{{ FullBody }}", context_json) # noqa: E501

    with http_client.client as session:
        response = _query_llm_score(
            prompt,
            session,
            azure_endpoint_url,
            token_manager,
            **request_args,
        )
        score = _post_process_retrieval_results(response)
    return score


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


def get_query_topic(row, good_topics_dict, bad_topics_dict, llm_summary_enabled):
    """Get the query topic."""
    idx = 0
    if len(good_topics_dict) == 0 and row[DOCUMENT_RELEVANCE_SCORE_COLUMN] >= GOOD_METRICS_VALUE:
        return DEFAULT_TOPIC_NAME
    if len(bad_topics_dict) == 0 and row[DOCUMENT_RELEVANCE_SCORE_COLUMN] < METRICS_VIOLATION_THRESHOLD:
        return DEFAULT_TOPIC_NAME

    for topic, q_list in bad_topics_dict.items():
        if row[MODIFIED_PROMPT_COLUMN] in q_list:
            # do not show the topic name when disabling the conf
            topic = topic if llm_summary_enabled == "true" else f"topic_{idx}"
            return topic
        idx += 1
    for topic, q_list in good_topics_dict.items():
        if row[MODIFIED_PROMPT_COLUMN] in q_list:
            # do not show the topic name when disabling the conf
            topic = topic if llm_summary_enabled == "true" else f"topic_{idx}"
            return topic
        idx += 1
    return None


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


def parse_debugging_info(root_span):
    """Parse the span tree to get debugging info."""
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        debugging_info_list = []
        prompt_flow_input = tree.root_span.input
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
                debugging_info = DebuggingInfo(span.span_id,
                                               index_content,
                                               index_id,
                                               query,
                                               TEXT_SPLITTER.join(text),
                                               max(score),
                                               retrieval_query_type,
                                               retrieval_top_k,
                                               prompt_flow_input)
                debugging_info_list.append(debugging_info.to_json_str())
        return debugging_info_list
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def expand_debugging_info(df):
    """Parse the debugging info and expand to span level."""
    df['debugging_info'] = df[ROOT_SPAN_COLUMN].apply(parse_debugging_info)
    df = df.explode('debugging_info')
    df[SPAN_ID_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["span_id"])
    df[INDEX_CONTENT_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["index_content"])
    df[INDEX_ID_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["index_id"])
    df[MODIFIED_PROMPT_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["query"])
    df[RETRIEVAL_DOC_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["retrieval_documents"])
    df[INDEX_SCORE_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["retrieval_score"])
    df[RETRIEVAL_QUERY_TYPE_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["retrieval_query_type"])
    df[RETRIEVAL_TOP_K_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["retrieval_top_k"])
    df[PROMPT_FLOW_INPUT_COLUMN] = df['debugging_info'].apply(lambda x: json.loads(x)["prompt_flow_input"])
    df = df.drop(columns='debugging_info')
    return df


def group_queries(df, workspace_connection_arm_id, model_deployment_name, llm_summary_enabled):
    """Get good and bad query group based on retrieval relevance score and assign topic."""
    bad_queries_df = df[df[DOCUMENT_RELEVANCE_SCORE_COLUMN] < METRICS_VIOLATION_THRESHOLD]
    good_queries_df = df[df[DOCUMENT_RELEVANCE_SCORE_COLUMN] >= GOOD_METRICS_VALUE]
    good_samples_df = good_queries_df.sample(n=min(len(bad_queries_df), len(good_queries_df)))
    bad_queries = bad_queries_df[MODIFIED_PROMPT_COLUMN].tolist()
    good_queries = good_samples_df[MODIFIED_PROMPT_COLUMN].tolist()

    if len(bad_queries) < GROUP_TOPIC_MIN_SAMPLE_SIZE:
        # Skip grouing if the sample size is too small
        print(f"Bad sample size {len(bad_queries)} is less than {GROUP_TOPIC_MIN_SAMPLE_SIZE}. Skip grouping and set default topic.")  # noqa
        bad_topics_dict = {}
    else:
        bad_topics_dict = bertopic_get_topic(bad_queries,
                                             workspace_connection_arm_id,
                                             model_deployment_name)
    if len(good_queries) < GROUP_TOPIC_MIN_SAMPLE_SIZE:
        # Skip grouing if the sample size is too small
        good_topics_dict = {}
    else:
        good_topics_dict = bertopic_get_topic(good_queries,
                                              workspace_connection_arm_id,
                                              model_deployment_name)
    df[QUERY_INTENTION_COLUMN] = df.apply(get_query_topic, axis=1, args=(good_topics_dict, bad_topics_dict, llm_summary_enabled,))  # noqa
    return df.dropna()


def perform_ttest(good_answer_scores, bad_answer_scores):
    """Perform Mann-Whitney U test."""
    t_stat, p_value = stats.ttest_ind(good_answer_scores, bad_answer_scores)
    print(f"Normal t-test T-statistic: {t_stat}, P-value: {p_value}")
    t_stat1, p_value1 = stats.ttest_ind(good_answer_scores, bad_answer_scores, equal_var=False)
    print(f"Welch's t-test T-statistic: {t_stat1}, P-value: {p_value1}")
    t_stat2, p_value2 = mannwhitneyu(good_answer_scores, bad_answer_scores, method='exact')
    print(f"Mann-Whitney U t-test T-statistic: {t_stat2}, P-value: {p_value2}")
    # Use Mann-Whitney U test for correlation test
    return t_stat2, p_value2


def peform_correlation_test(index, bad_answer_group, good_answer_group, query_intention):
    """Perform correlation test."""
    t_stat, p_value = perform_ttest(good_answer_group[INDEX_SCORE_LLM_COLUMN],
                                    bad_answer_group[INDEX_SCORE_LLM_COLUMN])
    bad_mean = statistics.mean(bad_answer_group[INDEX_SCORE_LLM_COLUMN])
    print("Mean value of bad group: ", bad_mean)
    if t_stat > 0 and p_value < P_VALUE_THRESHOLD and bad_mean < 3.0:
        print("Generating action.")
        return IndexAction(index, bad_answer_group[SPAN_ID_COLUMN].tolist(), query_intention, float(1.0 - p_value))
    return None


def merge_actions(action_1, action_2):
    """Merge actions."""
    if not action_1:
        return action_2
    if not action_2:
        return action_1
    query_ids = list(set(action_1.query_ids + action_1.query_ids))
    query_intention = ""
    confidence_score = 0
    if action_1.confidence_score > action_2.confidence_score:
        confidence_score = action_1.confidence_score
        query_intention = action_1.query_intention if action_1.query_intention != "default" else action_2.query_intention  # noqa
    else:
        confidence_score = action_2.confidence_score
        query_intention = action_2.query_intention if action_2.query_intention != "default" else action_1.query_intention  # noqa
    return Action("Index Action", query_ids, query_intention, confidence_score)


def generate_action(df, index_set):
    """Generate the action data."""
    actions = {}
    df = df[df[INDEX_SCORE_LLM_COLUMN] != DEFAULT_SCORE]
    for index in index_set:
        index_df = df[df[INDEX_ID_COLUMN] == index]
        bad_answer_df = index_df[index_df[DOCUMENT_RELEVANCE_SCORE_COLUMN] < METRICS_VIOLATION_THRESHOLD]
        good_answer_df = index_df[index_df[DOCUMENT_RELEVANCE_SCORE_COLUMN] >= GOOD_METRICS_VALUE]
        action_out = peform_correlation_test(index, bad_answer_df, good_answer_df, "default")
        query_intention_set = bad_answer_df[QUERY_INTENTION_COLUMN].unique()
        for query_intention in query_intention_set:
            bad_answer_group = bad_answer_df[bad_answer_df[QUERY_INTENTION_COLUMN] == query_intention]
            print("query:", query_intention)
            action = peform_correlation_test(index, bad_answer_group, good_answer_df, query_intention)
            merge_actions(action_out, action)
            if index not in actions:
                actions[index] = action
            else:
                actions[index] = merge_actions(actions[index], action)
        actions.append(action)
    return actions


def write_to_file(payload: dict, local_output_directory: str, file_name: str):
    """Save the action files to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    action_file = os.path.join(local_output_directory, f"{file_name}.json")
    with open(action_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


def is_index_asset(index_id):
    """Check if index id is asset id."""
    return index_id.startswith("azureml://")


def generate_samples(action_df, is_negative_sample):
    """Generate positive and negative samples in action file."""
    samples = []
    # sort the good samples by index score
    if not is_negative_sample:
        action_df = action_df.sort_values(by=INDEX_SCORE_LLM_COLUMN, ascending=False)
    for i in range(len(action_df.index)):
        if i >= MAX_SAMPLE_SIZE and not is_negative_sample:
            break
        sample = {
            "Question": action_df.iloc[i][PROMPT_COLUMN],
            "Answer": action_df.iloc[i][COMPLETION_COLUMN],
            "Topic": action_df.iloc[i][QUERY_INTENTION_COLUMN],
            "LookupScore": action_df.iloc[i][INDEX_SCORE_LLM_COLUMN],
            "DebuggingInfo": action_df.iloc[i][ROOT_SPAN_COLUMN],
            "RetrievalQueryType": action_df.iloc[i][RETRIEVAL_QUERY_TYPE_COLUMN],
            "RetrievalTopK": action_df.iloc[i][RETRIEVAL_TOP_K_COLUMN],
            "PromptFlowInput": action_df.iloc[i][PROMPT_FLOW_INPUT_COLUMN]
        }
        if is_negative_sample:
            sample["ViolatedMetrics"] = "Retrieval Relevance"
        samples.append(sample)
    return samples


def write_actions(df, actions, action_output_folder, aml_deployment_id):
    """Write the action summary and action detail files."""
    local_path = str(uuid.uuid4())
    action_summary = {}
    for index_id, action in actions.items():
        action_bad_df = df[(df[INDEX_ID_COLUMN] == index_id) & (df[SPAN_ID_COLUMN].isin(action.query_ids))]
        action_good_df = df[(df[INDEX_ID_COLUMN] == index_id) &
                            (df[DOCUMENT_RELEVANCE_SCORE_COLUMN] >= GOOD_METRICS_VALUE)]
        action_id = str(uuid.uuid4())
        action = {
            "ActionId": action_id,
            "Type": action.action_type,
            "Description": ACTION_DESCRIPTION.replace("{index_id}", index_id),
            "ConfidenceScore": action.confidence_score,
            "ViolatedMetrics": "Retrieval Relevance",
            "QueryIntention": action.query_intention,
            "CreationTime": str(datetime.datetime.now()),
            "FilePath": os.path.join(action_output_folder, f"actions/{action_id}.json")
        }
        action_summary[action_id] = action
        action_detail = copy.deepcopy(action)
        action_detail["DeploymentId"] = aml_deployment_id
        action_detail["RunId"] = os.environ.get("AZUREML_RUN_ID", None)
        if is_index_asset(index_id):
            action_detail["IndexAssetId"] = index_id
        else:
            action_detail["IndexName"] = index_id
        action_detail["IndexContent"] = action_bad_df.iloc[0][INDEX_CONTENT_COLUMN]
        action_detail["PositiveSamples"] = generate_samples(action_good_df, False)
        action_detail["NegativeSamples"] = generate_samples(action_bad_df, True)
        print("Writing action detail of action: ")
        print(action_detail)
        write_to_file(action_detail, local_path, action_id)
    print("Writing action summary to location ", action_output_folder)
    print(action_summary)
    write_to_file(action_summary, local_path, "action_summary")
    target_remote_path = os.path.join(action_output_folder, "actions")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)


def add_action_tag_to_run():
    """Add action analyzer tag for the root run."""
    root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID", None)
    print("Action generated, setting tag for pipeline run: ", root_run_id)
    client = MlflowClient()
    client.set_tag(root_run_id, "momo_action_analyzer_has_action", "true")


def relevance_action_detector(df,
                              workspace_connection_arm_id,
                              model_deployment_name,
                              llm_summary_enabled,
                              action_output_folder,
                              aml_deployment_id):
    """Relevance action detector."""
    # Expand to span level and get document relevance score
    df = expand_debugging_info(df)

    # Prepare llm variables
    request_args = get_llm_request_args(model_deployment_name)

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
        model_deployment_name
    )

    http_client = _HTTPClientWithRetry(
        n_retry=API_CALL_RETRY_MAX_COUNT,
        backoff_factor=API_CALL_RETRY_BACKOFF_FACTOR,
    )

    # get document relevance score
    df[DOCUMENT_RELEVANCE_SCORE_COLUMN] = df.apply(get_document_relevance_score, axis=1, args=(token_manager,
                                                                                               azure_endpoint_url,
                                                                                               http_client,
                                                                                               request_args,))
    print("Data with document relevance score.")
    print(df)

    # group the queries
    df = group_queries(df, workspace_connection_arm_id, model_deployment_name, llm_summary_enabled)
    print("Data with group.")
    print(df)

    # get retrieval score
    df[INDEX_SCORE_LLM_COLUMN] = df.apply(get_retrieval_score, axis=1, args=(token_manager,
                                                                             azure_endpoint_url,
                                                                             http_client,
                                                                             request_args,))
    print("Data with retrieval score.")
    print(df)

    # run t-test to generate action
    index_set = df[INDEX_ID_COLUMN].unique()
    actions = generate_action(df, index_set)
    if len(actions) == 0:
        print("No action generated.")
        return
    print(f"Get {len(actions)} actions")

    # write actions
    write_actions(df, actions, action_output_folder, aml_deployment_id)

    add_action_tag_to_run()
