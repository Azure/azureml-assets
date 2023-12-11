# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""Entry script for Model Performance Compute Metrics Spark Component."""

import argparse
import constants

from compute_metrics import EvaluatorFactory
from data_reader import DataReaderFactory
from utils import construct_signal_metrics



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
    parser.add_argument("--regression_rmse_threshold", type=str,
                        dest="regression_rmse_threshold", required=False, default=None)
    parser.add_argument("--regression_meanabserror_threshold", type=str,
                        dest="regression_meanabserror_threshold", required=False, default=None)
    parser.add_argument("--classification_precision_threshold", type=str,
                        dest="classification_precision_threshold", required=False, default=None)
    parser.add_argument("--classification_accuracy_threshold", type=str,
                        dest="classification_accuracy_threshold", required=False, default=None)
    parser.add_argument("--classification_recall_threshold", type=str,
                        dest="classification_recall_threshold", required=False, default=None)
    parser.add_argument("--signal_metrics", type=str)

    args = parser.parse_args()

    metrics_data = DataReaderFactory().get_reader(args.task).read_data(args.ground_truths,
                                                                       args.ground_truths_column_name,
                                                                       args.predictions,
                                                                       args.predictions_column_name)
    metrics = EvaluatorFactory().get_evaluator(task_type=args.task, metrics_config={}).evaluate(metrics_data)
    construct_signal_metrics(metrics,
                             args.signal_metrics,
                             args.regression_rmse_threshold,
                             args.regression_meanabserror_threshold,
                             args.classification_precision_threshold,
                             args.classification_accuracy_threshold,
                             args.classification_recall_threshold,
                             args.predictions_column_name
                             )


if __name__ == "__main__":
    run()
