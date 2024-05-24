# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Detector."""

import json
import argparse
import pandas
import traceback
from typing import List
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.action_detectors.action_detector import ActionDetector
from action_analyzer.contracts.action_detectors.low_retrieval_score_index_action_detector import (
    LowRetrievalScoreIndexActionDetector
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
    INDEX_ID_COLUMN
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
                    print("No look up span found, skip this span.")
                    continue
                index_span = tree.get_span_tree_node_by_span_id(parent_id)
                index_input = json.loads(index_span.input)
                index_content = index_input['mlindex_content']
                hashed_index_id = hash_index_content(index_content)
                if hashed_index_id:
                    index_list.append(hashed_index_id)
        return index_list if index_list != [] else None
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
    df.dropna(axis=0, subset=[INDEX_ID_COLUMN], inplace=True)
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
            average_score_name = f"Average{metrics}Score"
            if average_score_name in metrics_dict:
                if metrics_dict[average_score_name]["value"] < metrics_dict[average_score_name]["threshold"]:
                    print(f"Metrics {metrics} violated.")
                    violated_metrics.append(metrics)
        return violated_metrics
    except Exception as e:
        print("Exception while getting the violated metrics. ", e)
        return []


def run_detector(df: pandas.DataFrame,
                 detector: ActionDetector,
                 llm_client: LLMClient,
                 aml_deployment_id: str) -> List[Action]:
    """Run detector.

    Args:
        df(pandas.DataFrame): the input dataframe.
        detector(ActionDetector): the detector to run.
        llm_client(LLMClient): LLM client used to get some llm scores/info for action.
        aml_deployment_id(str): aml deployment id for the action.

    Returns:
        List[str]: list of violated metrics.
    """
    detector.preprocess_data(df, llm_client)
    return detector.detect(llm_client, aml_deployment_id)


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

    try:
        # get violated metrics
        violated_metrics = get_violated_metrics(args.signal_output, f"signals/{args.signal_name}")
        if violated_metrics == []:
            print("No violated metrics. No action will be generated.")
            return
        print("Violated metrics found: ", violated_metrics)

        # load scored data
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
            lrsi_detector = LowRetrievalScoreIndexActionDetector(index, violated_metrics, args.query_intention_enabled)
            index_actions += run_detector(df_pandas, lrsi_detector, llm_client, args.aml_deployment_id)

            # After all detectors, deduplicate actions if needed.
            final_actions += deduplicate_actions(index_actions)

        if len(final_actions) == 0:
            print("No action detected.")
            return
        print(f"{len(final_actions)} action(s) detected.")
        # write action files to output folder
        write_actions(final_actions, args.action_output)

        add_action_tag_to_run()
    except Exception as e:
        print("Action detector failed with exception:", e)
        print(traceback.format_exc())


if __name__ == "__main__":
    run()
