# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Detector."""

import json
import argparse
import pandas
from typing import List
from action_analyzer.contracts.action_detectors.low_retrieval_score_index_action_detector import (
    LowRetrievalScoreIndexActionDetector
)
from action_analyzer.contracts.llm_client import LLMClient
from action_analyzer.contracts.utils.detector_utils import (
    get_index_id_from_index_content,
    deduplicate_actions,
    write_actions,
    add_action_tag_to_run
)
from shared_utilities.constants import (
    RETRIEVAL_SPAN_TYPE,
    ROOT_SPAN_COLUMN,
    GSQ_METRICS_LIST,
    INDEX_ID_COLUMN
)
from shared_utilities.store_url import StoreUrl
from shared_utilities.io_utils import try_read_mltable_in_spark
from shared_utilities.span_tree_utils import SpanTree


def parse_index_id(root_span: str) -> List[str]:
    """Parse the span tree to get index id.

    Args:
        root_span(str): the span tree in json string format.

    Returns:
        list(str): list of parsed indexes in the span tree.
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
                print(index_span)
                index_input = json.loads(index_span.input)
                print(index_input)
                index_content = index_input['mlindex_content']
                index_id = get_index_id_from_index_content(index_content)
                if index_id:
                    index_list.append(index_id)
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
    df[INDEX_ID_COLUMN] = df[ROOT_SPAN_COLUMN].apply(parse_index_id)
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
        index_actions = []
        # low retrieval score index action detector
        lrsi_action_detector = LowRetrievalScoreIndexActionDetector(index,
                                                                    violated_metrics,
                                                                    args.query_intention_enabled)
        df_preprocessed = lrsi_action_detector.preprocess_data(df_pandas)
        if df_preprocessed:
            lrsi_actions = lrsi_action_detector.detect(df_preprocessed, llm_client, args.aml_deployment_id)
            index_actions += lrsi_actions

        # # Todo: metrics violation index action detector
        # mvi_action_detector = MetricsViolationIndexActionDetector()
        # df_preprocessed = mvi_action_detector.preprocess_data(df_pandas)
        # mvi_actions = mvi_action_detector.detect(df_preprocessed, llm_client, args.aml_deployment_id)
        # index_actions += mvi_actions

        # After all detectors, deduplicate actions if needed.
        final_actions += deduplicate_actions(index_actions)
    print(f"{len(final_actions)} actions are detected.")
    # write action files to output folder
    write_actions(final_actions, args.action_output)

    add_action_tag_to_run()


if __name__ == "__main__":
    run()
