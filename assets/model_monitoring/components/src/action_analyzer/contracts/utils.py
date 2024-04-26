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
from mlflow import MlflowClient
from shared_utilities.span_tree_utils import SpanTree
from shared_utilities.constants import (
    RETRIEVAL_SPAN_TYPE,
    GSQ_METRICS_LIST,
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
    DEFAULT_LLM_SCORE,
    ROOT_SPAN_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    PROMPT_COLUMN,
    DEFAULT_TOPIC_NAME,
    TEXT_SPLITTER
)
from shared_utilities.llm_utils import _APITokenManager, _request_api
from shared_utilities.io_utils import (
    np_encoder
)
from shared_utilities.amlfs import amlfs_upload
from shared_utilities.prompts import RELEVANCE_TEMPLATE, QUERY_INTENTION_PROMPT
from action_analyzer.contracts.index_action_sample import IndexActionSample
from action_analyzer.contract.actions import Action
from action_analyzer.contracts.llm_client import LLMClient


def convert_to_camel_case(input_string: str) -> str:
    """
    Convert a snake_case string to camelCase.

    Example: "retrieval_top_k" -> "RetrievalTopK"
    """
    words = input_string.split("_")
    result = "".join(word.capitalize() for word in words)
    return result


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


def parse_debugging_info(root_span: str, detector_index_id: str) -> list[str]:
    """Parse the span tree to get debugging info.

    Args:
        root_span(str): the span tree in json string format.
        detector_index_id(str): the index id specific to the detector.

    Returns:
        list[str]: list of extra debugging fields in serilized dictionary.
    """
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        extra_feilds_list = []
        prompt_flow_input = tree.root_span.input
        for span in tree:
            if span.span_type == RETRIEVAL_SPAN_TYPE:
                parent_id = span.parent_id
                if not parent_id:
                    print("No look up span found, skip this span.")
                    continue
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
                index_input = json.loads(json.loads(index_span.attributes)["inputs"])
                index_content = index_input['mlindex_content']
                index_id = get_index_id_from_index_content(index_content)
                # only get the data for the detector index id
                if index_id != detector_index_id:
                    continue
                retrieval_query_type = index_input["query_type"]
                retrieval_top_k = index_input["top_k"]
                retrieval_info = json.loads(span.attributes)
                query = retrieval_info["retrieval.query"]
                retrieval_documents = json.loads(retrieval_info["retrieval.documents"])
                texts = []
                scores = []
                for document in retrieval_documents:
                    texts.append(document["document.content"])
                    scores.append(float(document["document.score"]))
                extra_fields = {
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
                extra_feilds_list.append(json.dumps(extra_fields))
        return extra_feilds_list if extra_feilds_list != [] else None
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def extract_fields_from_debugging_info(df: pandas.DataFrame, detector_index_id: str) -> pandas.DataFrame:
    """Parse the debugging info, expand the dataframe to span level, and populate extra fields as new columns.

    Args:
        df(pandas.DataFrame): input dataframe in trace level.
        detector_index_id(str): the index id specific to the detector.

    Returns:
        pandas.DataFrame: dataframe with enriched debugging values in span level.
    """
    df['extra_fields'] = df[ROOT_SPAN_COLUMN].apply(parse_debugging_info, args=(detector_index_id,))
    # filter rows only for the detector index id
    df.dropna(subset=['extra_fields'], inplace=True)
    df = df.explode('extra_fields')
    df[SPAN_ID_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[SPAN_ID_COLUMN])
    df[INDEX_CONTENT_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[INDEX_CONTENT_COLUMN])
    df[MODIFIED_PROMPT_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[MODIFIED_PROMPT_COLUMN])
    df[RETRIEVAL_DOC_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[RETRIEVAL_DOC_COLUMN])
    df[INDEX_SCORE_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[INDEX_SCORE_COLUMN])
    df[RETRIEVAL_QUERY_TYPE_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[RETRIEVAL_QUERY_TYPE_COLUMN])
    df[RETRIEVAL_TOP_K_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[RETRIEVAL_TOP_K_COLUMN])
    df[PROMPT_FLOW_INPUT_COLUMN] = df['extra_fields'].apply(lambda x: json.loads(x)[PROMPT_FLOW_INPUT_COLUMN])
    df = df.drop(columns='extra_fields')
    return df


def get_missed_metrics(violated_metrics: list[str], column_names: list[str]) -> list[str]:
    """Get missed e2e metrics in the input dataframe.

    Args:
        violated_metrics(list[str]): list of violated metrics.
        column_names(list[str]): list of column names in dataframe.

    Returns:
        list[str]: list of missed e2e metrics.
    """
    missed_metrics = []
    for metric in violated_metrics:
        # skip the invalid metrics for now.
        if metric not in GSQ_METRICS_LIST:
            print(f"[Warning]Metric {metric} is not supported.")
            continue
        if metric not in column_names:
            print(f"Metirc {metric} is not in the input dataframe.")
            missed_metrics.append(metric)
    return missed_metrics


def calculate_e2e_metrics(df: pandas.DataFrame, missed_metrics: list[str]) -> pandas.DataFrame:
    """[Todo] Calculate missed e2e metrics.

    Args:
        df(pandas.DataFrame): input dataframe.
        missed_metrics(list[str]): list of missed e2e metrics to be calculated.

    Returns:
        pandas.DataFrame: dataframe with calculated metrics.
    """
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
        score = DEFAULT_LLM_SCORE
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


def get_query_intention(query_list: list[str], llm_client: LLMClient) -> str:
    """Get the query intention from a list of queries.

    Args:
        query_list(list[str]): list of queries.
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
                                  is_negative_sample: bool,
                                  max_positive_sample_size: int):
    """Generate index action samples.

    Args:
        df(pandas.DataFrame): dataframe for generating action samples.
        is_negative_sample(bool): flag to show if the sample is negative.
        max_positive_sample_size(int): max positive sample size in the action.

    Returns:
        pandas.DataFrame: dataframe with calculated metrics.
    """
    samples = []
    # sort the positive samples by index score
    if not is_negative_sample:
        df = df.sort_values(by=INDEX_SCORE_LLM_COLUMN, ascending=False)
    for i in range(len(df.index)):
        if i >= max_positive_sample_size and not is_negative_sample:
            break
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


def deduplicate_actions(actions: list[Action]) -> list[Action]:
    """Deduplicate actions.

    Args:
        actions(list[Action]): list of actions.

    Returns:
        list[Action]: list of actions after deduplication.
    """
    # Todo: do this later
    return actions


def write_to_file(payload: dict, local_output_directory: str, file_name: str):
    """Write a file to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    action_file = os.path.join(local_output_directory, f"{file_name}.json")
    with open(action_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


def generate_action_summary(actions: list[Action], action_output_folder: str) -> dict:
    """Genearte action summary file from the action list.

    Args:
        actions(list[Action]): list of actions.
        action_output_folder(str): action output folder path.

    Returns:
        dict: action summary file in dict format. Key is action id, value is action data.
    """
    action_summary = {}
    for action in actions:
        action_summary[action.action_id] = action.to_summary_json(action_output_folder)
    return action_summary


def write_actions(actions: list[Action], action_output_folder: str) -> None:
    """Write action summary and action detail files.

    Args:
        actions(list[Action]): list of actions.
        action_output_folder(str): output folder path.
    """
    local_path = str(uuid.uuid4())
    action_summary = generate_action_summary(actions)
    for action in actions:
        write_to_file(action.to_json(), local_path, action.action_id)
    print("Writing action summary to location ", action_output_folder)
    print(action_summary)
    write_to_file(action_summary, local_path, "action_summary")
    target_remote_path = os.path.join(action_output_folder, "actions")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)


def add_action_tag_to_run() -> None:
    """Add action analyzer tag for the root run."""
    root_run_id = os.environ.get("AZUREML_ROOT_RUN_ID", None)
    print("Action generated, setting tag for pipeline run: ", root_run_id)
    client = MlflowClient()
    client.set_tag(root_run_id, "momo_action_analyzer_has_action", "true")
