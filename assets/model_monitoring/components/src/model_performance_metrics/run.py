# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse

from shared_utilities.patch_mltable import patch_all
import constants

from data_reader import DataReaderFactory
from compute_metrics import EvaluatorFactory
from utils import write_to_mltable

patch_all()


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, dest="task", choices=constants.ALL_TASKS)
    parser.add_argument("--baseline_data", type=str, dest="ground_truths", required=True)
    parser.add_argument("--baseline_data_column_name", type=str,
                        dest="ground_truths_column_name", required=False, default=None)
    parser.add_argument("--production_data", type=str, dest="predictions", required=True)
    parser.add_argument("--production_data_column_name", type=str,
                        dest="predictions_column_name", required=False, default=None)
    parser.add_argument("--signal_metrics", type=str)
    args = parser.parse_args()

    metrics_data = DataReaderFactory().get_reader(args.task).read_data(args.ground_truths,
                                                                       args.ground_truths_column_name, args.predictions,
                                                                       args.predictions_column_name)
    metrices = EvaluatorFactory().get_evaluator(task_type=args.task, metrics_config={}).evaluate(metrics_data)
    write_to_mltable(metrices, args.signal_metrics)


if __name__ == "__main__":
    run()
