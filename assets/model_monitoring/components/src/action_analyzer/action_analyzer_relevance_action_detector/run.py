# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Relevance Action Detector."""

import argparse
from action_analyzer.action_analyzer_relevance_action_detector.relevance_action_detector import relevance_action_detector

def run():
    """Relevance action detector."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--signal_scored_data", type=str)
    parser.add_argument("--model_deployment_name", type=str, required=True)
    parser.add_argument("--workspace_connection_arm_id", type=str, required=True)
    parser.add_argument("--aml_deployment_id", type=str)
    parser.add_argument("--llm_summary_enabled", type=str)
    args = parser.parse_args()

    signal_scored_data_df = try_read_mltable_in_spark("azureml://subscriptions/1aefdc5e-3a7c-4d71-a9f9-f5d3b03be19a/resourcegroups/yuachengtestrg/workspaces/ai-proj-eastus/datastores/workspaceblobstore/paths/azureml/fd174990-a0e4-4447-a553-387391889682/evaluation/", "signal_scored_data")
    print("gsq output df")
    signal_scored_data_df.show()

    df_pandas = signal_scored_data_df.toPandas()

    relevance_action_detector(df_pandas, args.workspace_connection_arm_id, args.model_deployment_name, args.llm_summary_enabled, args.action_output, args.llm_summary_enabled)

if __name__ == "__main__":
    run()
