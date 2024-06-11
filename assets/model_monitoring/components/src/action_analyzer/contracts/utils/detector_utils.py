# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities for action detector."""
import os
import uuid
import json
import pandas
import yaml
import requests
import re
import hashlib
from scipy import stats
from scipy.stats import mannwhitneyu
from typing import List
from mlflow import MlflowClient
from assets.model_monitoring.components.src.shared_utilities.store_url import StoreUrl
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.constants import (
    RETRIEVAL_SPAN_TYPE,
    MODIFIED_PROMPT_COLUMN,
    COMPLETION_COLUMN,
    RETRIEVAL_DOC_COLUMN,
    SPAN_ID_COLUMN,
    INDEX_CONTENT_COLUMN,
    INDEX_ID_COLUMN,
    INDEX_SCORE_COLUMN,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    PROMPT_FLOW_INPUT_COLUMN,
    INVALID_LLM_SCORE,
    ROOT_SPAN_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    PROMPT_COLUMN,
    DEFAULT_TOPIC_NAME,
    TEXT_SPLITTER,
    MAX_SAMPLE_SIZE,
    TTEST_NAME
)
from shared_utilities.llm_utils import _APITokenManager, _request_api
from shared_utilities.io_utils import (
    np_encoder
)
from shared_utilities.prompts import RELEVANCE_TEMPLATE, QUERY_INTENTION_PROMPT
from action_analyzer.contracts.action_sample import IndexActionSample
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient


def hash_index_content(index_content: str) -> str:
    """Convert index content to sha256 hash string.

    Args:
        index_content(str): str of index yaml content.

    Returns:
        str: the sha256 hash string.
    """
    return hashlib.sha256(index_content.encode('utf-8')).hexdigest()


def get_index_id_from_index_content(index_content: str) -> str:
    """Parse the index id from index yaml.

    Args:
        index_content(str): str of index yaml content.

    Returns:
        str: the index id parsed from the index yaml content.
    """
    index_payload = yaml.safe_load(index_content)
    # if the asset id does not exist, use the index name
    if "self" in index_payload:
        index_id = index_payload["self"].get("asset_id", None)
    elif "index" in index_payload:
        index_id = index_payload["index"].get("index", None)
    else:
        index_id = None
    return index_id


def extract_retrieval_info(root_span: str, detector_index_id: str) -> List[str]:
    """Parse the span tree to get retrieval information. Only for detector hashed index id.

    Args:
        root_span(str): the span tree in json string format.
        detector_index_id(str): the hashed index id specific to the detector.

    Returns:
        List[str]: list of extra debugging fields in serilized dictionary.
    """
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        retrieval_info_list = []
        prompt_flow_input = tree.root_span.input
        for span in tree:
            if span.span_type == RETRIEVAL_SPAN_TYPE:
                parent_id = span.parent_id
                if not parent_id:
                    print("No look up span found, skip this span.")
                    continue
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
                index_input = json.loads(index_span.input)
                index_content = index_input['mlindex_content']
                hashed_index_id = hash_index_content(index_content)
                # only get the data for the detector hashed index id
                if hashed_index_id != detector_index_id:
                    continue
                # index asset id is only used for rag debugging
                index_id = get_index_id_from_index_content(index_content)
                retrieval_query_type = index_input["query_type"]
                retrieval_top_k = index_input["top_k"]
                retrieval_attributes = json.loads(span.attributes)
                query = retrieval_attributes["retrieval.query"]
                retrieval_documents = json.loads(retrieval_attributes["retrieval.documents"])
                texts = []
                scores = []
                for document in retrieval_documents:
                    texts.append(document["document.content"])
                    scores.append(float(document["document.score"]))
                retrieval_info = {
                    SPAN_ID_COLUMN: span.span_id,
                    INDEX_CONTENT_COLUMN: index_content,
                    INDEX_ID_COLUMN: index_id,
                    MODIFIED_PROMPT_COLUMN: query,
                    RETRIEVAL_DOC_COLUMN: TEXT_SPLITTER.join(texts),
                    INDEX_SCORE_COLUMN: max(scores),
                    RETRIEVAL_QUERY_TYPE_COLUMN: retrieval_query_type,
                    RETRIEVAL_TOP_K_COLUMN: retrieval_top_k,
                    PROMPT_FLOW_INPUT_COLUMN: prompt_flow_input
                }
                retrieval_info_list.append(json.dumps(retrieval_info))
        return retrieval_info_list if retrieval_info_list != [] else None
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def extract_retrieval_info_from_root_span(df: pandas.DataFrame, detector_index_id: str) -> pandas.DataFrame:
    """Parse the root span, expand the dataframe to span level, and populate retrieval info fields as new columns.

    Args:
        df(pandas.DataFrame): input dataframe in trace level.
        detector_index_id(str): the hashed index id specific to the detector.

    Returns:
        pandas.DataFrame: dataframe with enriched debugging values in span level.
    """
    df['retrieval_info'] = df[ROOT_SPAN_COLUMN].apply(extract_retrieval_info, args=(detector_index_id,))
    # filter rows only for the detector index id
    df.dropna(subset=['retrieval_info'], inplace=True)
    df = df.explode('retrieval_info')
    df[SPAN_ID_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[SPAN_ID_COLUMN])
    df[INDEX_ID_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[INDEX_ID_COLUMN])
    df[INDEX_CONTENT_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[INDEX_CONTENT_COLUMN])
    df[MODIFIED_PROMPT_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[MODIFIED_PROMPT_COLUMN])
    df[RETRIEVAL_DOC_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[RETRIEVAL_DOC_COLUMN])
    df[INDEX_SCORE_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[INDEX_SCORE_COLUMN])
    df[RETRIEVAL_QUERY_TYPE_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[RETRIEVAL_QUERY_TYPE_COLUMN])
    df[RETRIEVAL_TOP_K_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[RETRIEVAL_TOP_K_COLUMN])
    df[PROMPT_FLOW_INPUT_COLUMN] = df['retrieval_info'].apply(lambda x: json.loads(x)[PROMPT_FLOW_INPUT_COLUMN])
    df = df.drop(columns='retrieval_info')
    return df


def _prepare_retrieval_template_inputs(row):
    question = row[MODIFIED_PROMPT_COLUMN]
    answer = row[COMPLETION_COLUMN]
    text = row[RETRIEVAL_DOC_COLUMN]

    context_array = text.split(TEXT_SPLITTER)
    context_json = {}
    for i, context in enumerate(context_array):
        context_json[f"Document {i}"] = context
    return (question, answer, json.dumps(context_json))


def _post_process_retrieval_results(response):
    output = response["samples"][0]
    parsed_score_response = re.findall(r'\d+', output.split("# Result")[-1].strip())
    if len(parsed_score_response) > 0:
        score = float(parsed_score_response[0].replace("'", "").strip())
    else:
        # Result of score is not found in the output string
        score = INVALID_LLM_SCORE
        print("Not able to parse the retrieval score, setting score to 0")
    return score


def _post_process_query_intention_results(response):
    output = response["samples"][0]
    parsed_query_intention = output.split("Topic name:")[-1].strip()
    if len(parsed_query_intention) > 0:
        query_intention = parsed_query_intention
    else:
        # Result of query intention is not found. Use the default topic.
        query_intention = DEFAULT_TOPIC_NAME
    return query_intention


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


def get_retrieval_score(row: pandas.Series,
                        llm_client: LLMClient) -> int:
    """Get retrieval score using llm.

    Args:
        row(pandas.Series): row of dataframe.
        llm_client(LLM): llm client to make llm call.

    Returns:
        int: retrieval score generated by llm.
    """
    question, answer, context_json = _prepare_retrieval_template_inputs(row)
    prompt = RELEVANCE_TEMPLATE.replace("{{ query }}", question).replace("{{ history }}", "").replace("{{ FullBody }}", context_json)  # noqa: E501

    with llm_client.http_client.client as session:
        response = _query_llm_score(
            prompt,
            session,
            llm_client.azure_endpoint_url,
            llm_client.token_manager,
            **(llm_client.llm_request_args),
        )
        score = _post_process_retrieval_results(response)
    return score


def get_query_intention(query_list: List[str], llm_client: LLMClient) -> str:
    """Get the query intention from a list of queries.

    Args:
        query_list(List[str]): list of queries.
        llm_client(LLM): llm client to make llm call.

    Returns:
        str: the summarized query intention for the queries.
    """
    queries = "- " + "\n- ".join(query_list)
    prompt = QUERY_INTENTION_PROMPT.replace("{{ QUERIES }}", queries)

    with llm_client.http_client.client as session:
        response = _query_llm_score(
            prompt,
            session,
            llm_client.azure_endpoint_url,
            llm_client.token_manager,
            **(llm_client.llm_request_args),
        )
        query_intention = _post_process_query_intention_results(response)
    return query_intention


def generate_index_action_samples(df: pandas.DataFrame,
                                  is_negative_sample: bool):
    """Generate index action samples.

    Args:
        df(pandas.DataFrame): dataframe for generating action samples.
        is_negative_sample(bool): flag to show if the sample is negative.

    Returns:
        pandas.DataFrame: dataframe with calculated metrics.
    """
    samples = []
    # sort the positive samples by index score
    if not is_negative_sample:
        df = df.sort_values(by=INDEX_SCORE_LLM_COLUMN, ascending=False)
    for i in range(len(df.index)):
        index_sample = IndexActionSample(
            df.iloc[i][PROMPT_COLUMN],
            df.iloc[i][MODIFIED_PROMPT_COLUMN],
            df.iloc[i][COMPLETION_COLUMN],
            df.iloc[i][INDEX_SCORE_LLM_COLUMN],
            df.iloc[i][ROOT_SPAN_COLUMN],
            df.iloc[i][RETRIEVAL_QUERY_TYPE_COLUMN],
            df.iloc[i][RETRIEVAL_TOP_K_COLUMN],
            df.iloc[i][PROMPT_FLOW_INPUT_COLUMN]
        )
        samples.append(index_sample)
    return samples


def perform_ttest(good_group_retrieval_scores: pandas.Series,
                  bad_group_retrieval_scores: pandas.Series) -> (float, float):
    """Perform Mann-Whitney U test.

    Args:
        good_group_retrieval_scores(pandas.Series): retrieval scores of high metric score group.
        bad_group_retrieval_scores(pandas.Series): retrieval scores of low metric score group.

    Returns:
        float: test statistic calculated in t-test.
        float: p-value calcualted in t-test
    """
    t_stat, p_value = stats.ttest_ind(good_group_retrieval_scores, bad_group_retrieval_scores)
    print(f"Normal t-test T-statistic: {t_stat}, P-value: {p_value}")
    t_stat1, p_value1 = stats.ttest_ind(good_group_retrieval_scores, bad_group_retrieval_scores, equal_var=False)
    print(f"Welch's t-test T-statistic: {t_stat1}, P-value: {p_value1}")
    t_stat2, p_value2 = mannwhitneyu(good_group_retrieval_scores, bad_group_retrieval_scores, method='exact')
    print(f"Mann-Whitney U t-test T-statistic: {t_stat2}, P-value: {p_value2}")
    # Use Mann-Whitney U test for correlation test
    return t_stat2, p_value2


def peform_correlation_test(high_metric_score_df: pandas.DataFrame,
                            low_metric_score_df: pandas.DataFrame,
                            correlation_test_method: str) -> (float, float):
    """Perform correlation test.

    Args:
        high_metric_score_df(pandas.DataFrame): dataframe of queires with high metric score.
        low_metric_score_df(pandas.DataFrame): dataframe of queries with low metric score.
        correlation_test_method(str): correlation test method. Default to t-test.

    Returns:
        float: test statistic calculated in t-test.
        float: p-value calcualted in t-test
    """
    # only support t-test for now
    if correlation_test_method == TTEST_NAME:
        t_stat, p_value = perform_ttest(high_metric_score_df[INDEX_SCORE_LLM_COLUMN],
                                        low_metric_score_df[INDEX_SCORE_LLM_COLUMN])
        return (t_stat, p_value)
    return (-1, -1)


def calculate_action_overlap(action1: Action, action2: Action) -> float:
    """Get the action overlap rate of 2 actions.

    Args:
        action1(Action): action 1.
        action2(Action): action 2.

    Returns:
        float: the action overlap rate.
    """
    negative_samples_1 = [sample.question for sample in action1.negative_samples]
    negative_samples_2 = [sample.question for sample in action2.negative_samples]
    intersection = len(set(negative_samples_1) & set(negative_samples_2))
    union = len(set(negative_samples_1) | set(negative_samples_2))
    return intersection / union


def deduplicate_actions(action_list: List[Action]) -> List[Action]:
    """Deduplicate actions.

    Args:
        action_list(List[Action]): list of actions.

    Returns:
        List[Action]: list of actions after deduplication.
    """
    sorted_actions = sorted(action_list, key=lambda x: x.confidence_score, reverse=True)
    deduplicated_list = []
    for action in sorted_actions:
        is_unique_action = True
        for unique_action in deduplicated_list:
            overlap = calculate_action_overlap(action, unique_action)
            if overlap > 0.5:
                # do not add this action.
                is_unique_action = False
                break
        if is_unique_action:
            deduplicated_list.append(action)
    print(f"Deduplicated from {len(action_list)} actions to {len(deduplicated_list)}.")
    return deduplicated_list


def write_to_file(payload: dict, store_url: StoreUrl, file_name: str):
    """Write a file to a store_url file."""
    content = json.dumps(payload, indent=4, default=np_encoder)
    store_url.write_file(content, file_name, True)


def generate_action_summary(actions: List[Action], action_output_folder: str) -> dict:
    """Genearte action summary file from the action list.

    Args:
        actions(List[Action]): list of actions.
        action_output_folder(str): action output folder path.

    Returns:
        dict: action summary file in dict format. Key is action id, value is action data.
    """
    action_summary = {}
    for action in actions:
        action_summary[action.action_id] = action.to_summary_json(action_output_folder)
    return action_summary


def write_actions(actions: List[Action], action_output_folder: str) -> None:
    """Write action summary and action detail files.

    Args:
        actions(List[Action]): list of actions.
        action_output_folder(str): output folder path.
    """
    target_remote_path = os.path.join(action_output_folder, "actions")
    store_url = StoreUrl(target_remote_path)
    action_summary = generate_action_summary(actions, action_output_folder)
    for action in actions:
        action.reduce_positive_sample_size(MAX_SAMPLE_SIZE)
        write_to_file(action.to_json(), store_url, f"{action.action_id}.json")
    print("Writing action summary to location ", action_output_folder)
    print(action_summary)
    write_to_file(action_summary, store_url, "action_summary.json")


def add_action_tag_to_run() -> None:
    """Add action analyzer tag for the root run."""
    root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID", None)
    print("Action generated, setting tag for pipeline run: ", root_run_id)
    client = MlflowClient()
    client.set_tag(root_run_id, "momo_action_analyzer_has_action", "true")
