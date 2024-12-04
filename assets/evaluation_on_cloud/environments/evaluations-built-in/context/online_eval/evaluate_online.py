# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main script for the online evaluation context."""
import argparse
import preprocess
import evaluate
import postprocess
import mlflow


def get_args():
    """Get arguments from the command line."""
    parser = argparse.ArgumentParser()
    # Inputs
    parser.add_argument("--connection_string", type=str, dest="connection_string")
    parser.add_argument("--resource_id", type=str, dest="resource_id")
    parser.add_argument("--query", type=str, dest="query")
    parser.add_argument("--sampling_rate", type=str, dest="sampling_rate", default="1")
    parser.add_argument("--preprocessor_connection_type", type=str, dest="preprocessor_connection_type",
                        default="user-identity")
    parser.add_argument("--cron_expression", type=str, dest="cron_expression", default="0 0 * * *")
    parser.add_argument("--preprocessed_data", type=str, dest="preprocessed_data",
                        default="./preprocessed_data_output.jsonl")
    parser.add_argument("--evaluated_data", type=str, dest="evaluated_data", default="./evaluated_data_output.jsonl")
    parser.add_argument("--evaluators", type=str, dest="evaluators")
    parser.add_argument("--evaluator_name_id_map", type=str, dest="evaluator_name_id_map")
    parser.add_argument("--service_name", type=str, dest="service_name", default=None)

    args, _ = parser.parse_known_args()
    return vars(args)


def run():
    """Entry point for the script."""
    args = get_args()
    preprocess.run(args)
    evaluate.run(args)
    postprocess.run(args)

    # Evaluate


if __name__ == "__main__":
    with mlflow.start_run() as _run:
        run()
