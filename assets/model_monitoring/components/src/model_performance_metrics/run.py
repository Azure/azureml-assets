# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
import constants

from compute_metrics import EvaluatorFactory
from data_reader import DataReaderFactory
from utils import write_to_mltable


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS)
    parser.add_argument("--baseline_data", type=str, dest="ground_truths", required=True)
    parser.add_argument("--baseline_data_target_column", type=str,
                        dest="ground_truths_column_name", required=False, default=None)
    parser.add_argument("--production_data", type=str, dest="predictions", required=True)
    parser.add_argument("--production_data_target_column", type=str,
                        dest="predictions_column_name", required=False, default=None)
    parser.add_argument("--signal_metrics", type=str)
    args = parser.parse_args()

    metrics_data = DataReaderFactory().get_reader(args.task).read_data(args.ground_truths,
                                                                       args.ground_truths_column_name,
                                                                       args.predictions,
                                                                       args.predictions_column_name)
    metrics = EvaluatorFactory().get_evaluator(task_type=args.task, metrics_config={}).evaluate(metrics_data)
    write_to_mltable(metrics, args.signal_metrics)


if __name__ == "__main__":
    run()
