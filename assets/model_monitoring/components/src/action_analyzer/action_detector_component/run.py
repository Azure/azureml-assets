# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Detector."""

import json
import argparse
import pandas
from typing import List
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.action_detectors.action_detector import ActionDetector
from action_analyzer.contracts.action_detectors.low_retrieval_score_index_action_detector import (
    LowRetrievalScoreIndexActionDetector
)
from action_analyzer.contracts.action_detectors.metrics_violation_index_action_detector import (
    MetricsViolationIndexActionDetector
)
from action_analyzer.contracts.llm_client import LLMClient
from action_analyzer.contracts.utils.detector_utils import (
    deduplicate_actions,
    write_actions,
    add_action_tag_to_run,
    hash_index_content
)
from shared_utilities.constants import (
    RETRIEVAL_SPAN_TYPE,
    ROOT_SPAN_COLUMN,
    GSQ_METRICS_LIST,
    INDEX_ID_COLUMN,
    P_VALUE_THRESHOLD,
    TTEST_NAME
)
from shared_utilities.store_url import StoreUrl
from shared_utilities.io_utils import try_read_mltable_in_spark
from shared_utilities.span_tree_utils import SpanTree


def parse_hashed_index_id(root_span: str) -> List[str]:
    """Parse the span tree to get hashed index id.

    Args:
        root_span(str): the span tree in json string format.

    Returns:
        list(str): list of hashed indexes in the span tree.
    """
    try:
        tree = SpanTree.create_tree_from_json_string(root_span)
        index_list = []
        for span in tree:
            if span.span_type == RETRIEVAL_SPAN_TYPE:
                parent_id = span.parent_id
                if not parent_id:
                    print("No look up span found, skip action analyzer.")
                    return None
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
                index_input = json.loads(index_span.input)
                index_content = index_input['mlindex_content']
                hashed_index_id = hash_index_content(index_content)
                if hashed_index_id:
                    index_list.append(hashed_index_id)
        return index_list
    except KeyError as e:
        print("Required field not found: ", e)
        return None


def get_unique_indexes(df: pandas.DataFrame) -> List[str]:
    """Parse the span tree to find all unique indexes.

    Args:
        df(pandas.DataFrame): input pandas dataframe in trace level.

    Returns:
        List[str]: list of unique indexes.
    """
    df[INDEX_ID_COLUMN] = df[ROOT_SPAN_COLUMN].apply(parse_hashed_index_id)
    # expand each trace row to index level
    df = df.explode(INDEX_ID_COLUMN)
    return df[INDEX_ID_COLUMN].unique().tolist()


def get_violated_metrics(signal_out_url: str, signal_name: str) -> List[str]:
    """Get the violated metrics names from the gsq output.

    Args:
        signal_out_url(str): gsq output url.
        signal_name(str): signal name defined by user.

    Returns:
        List[str]: list of violated metrics.
    """
    violated_metrics = []
    try:
        store_url = StoreUrl(signal_out_url)
        gsq_output = store_url.read_file_content(f"{signal_name}.json")
        gsq_output_json = json.loads(gsq_output)
        metrics_dict = gsq_output_json["metrics"]
        for metrics in GSQ_METRICS_LIST:
            pass_rate_metrics = f"Average{metrics}Score"
            if pass_rate_metrics in metrics_dict:
                if metrics_dict[pass_rate_metrics]["value"] < metrics_dict[pass_rate_metrics]["threshold"]:
                    print(f"Metrics {metrics} violated.")
                    violated_metrics.append(metrics)
        return violated_metrics
    except Exception as e:
        print("Exception while getting the violated metrics. ", e)
        return []


def create_detectors(**params: dict) -> List[ActionDetector]:
    """Initialize all supported detectors.

    Args:
        params(dict): parameters for detectors.

    Returns:
        List[ActionDetector]: list of violated metrics.
    """
    detectors = []
    detectors.append(LowRetrievalScoreIndexActionDetector(params["hashed_index"],
                                                          params["violated_metrics"],
                                                          params["query_intention_enabled"]))
    detectors.append(MetricsViolationIndexActionDetector(params["hashed_index"],
                                                         params["violated_metrics"],
                                                         TTEST_NAME,
                                                         P_VALUE_THRESHOLD,
                                                         params["query_intention_enabled"]))
    return detectors


def run_detectors(df: pandas.DataFrame, detectors: List[ActionDetector], **extra_params: dict) -> List[Action]:
    """Run all detectors.

    Args:
        df(pandas.DataFrame): input pandas dataframe.
        detectors(List[ActionDetector]): list of available detectors.
        extra_params(dict): parameters for running detectors.

    Returns:
        List[str]: list of violated metrics.
    """
    action_list = []
    for detector in detectors:
        df_preprocessed = detector.preprocess_data(df)
        if not df_preprocessed.empty:
            action_list += detector.detect(df_preprocessed,
                                           extra_params["llm_client"],
                                           extra_params["aml_deployment_id"])
    return action_list


def run():
    """Script for action detector component."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--signal_scored_data", type=str)
    parser.add_argument("--signal_output", type=str)
    parser.add_argument("--signal_name", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    parser.add_argument("--aml_deployment_id", type=str)
    parser.add_argument("--query_intention_enabled", type=str)
    args = parser.parse_args()

    # get violated metrics
    violated_metrics = get_violated_metrics(args.signal_output, f"signals/{args.signal_name}")
    if violated_metrics == []:
        print("No violated metrics. No action will be generated.")
        return
    print("Violated metrics found: ", violated_metrics)

    signal_scored_data_df = try_read_mltable_in_spark(args.signal_scored_data, "signal_scored_data")
    print("gsq output df")
    signal_scored_data_df.show()
    df_pandas = signal_scored_data_df.toPandas()

    # find unique indexes
    unique_indexes = get_unique_indexes(df_pandas)
    if not unique_indexes or len(unique_indexes) == 0:
        print("No index found in the data input. No action will be generated.")
        return
    print(f"Found {len(unique_indexes)} indexes in the input data.")

    llm_client = LLMClient(args.workspace_connection_arm_id, args.model_deployment_name)
    # detect actions for each index
    final_actions = []
    for index in unique_indexes:
        detectors = create_detectors(hashed_index=index,
                                     violated_metrics=violated_metrics,
                                     query_intention_enabled=args.query_intention_enabled)

        index_actions = run_detectors(df=df_pandas,
                                      detectors=detectors,
                                      llm_client=llm_client,
                                      aml_deployment_id=args.aml_deployment_id)

        # After all detectors, deduplicate actions if needed.
        final_actions += deduplicate_actions(index_actions)

    if len(final_actions) == 0:
        print("No action detected.")
        return
    print(f"{len(final_actions)} action(s) detected.")
    # write action files to output folder
    write_actions(final_actions, args.action_output)

    add_action_tag_to_run()


if __name__ == "__main__":
    run()