# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import logging

from responsibleai.model_monitor.components.src.feature_importance_utilities import get_model_wrapper, compute_explanations,  compute_categorical_features
from io_utils import load_mltable_to_df

from tabular.components.src._telemetry._loggerfactory import _LoggerFactory, track

_logger = logging.getLogger(__file__)
_ai_logger = None


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = _LoggerFactory.get_logger(__file__)
    return _ai_logger


_get_logger()


def parse_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str)

    args = parser.parse_args()

    return args

def compute_feature_importance(task_type, target_column, baseline_dataframe):
    """Compute feature importance of baseline dataframe
      :param task_type: The task type (regression or classification) of the resulting model
      :type task_type: string
      :param target_column: the column to predict
      :type target_column: string
      :param baseline_dataframe: The baseline data meaning the data used to create the
      model monitor
      :type baseline_dataframe: pandas.DataFrame
      :return: list of feature importances in the order of the columns in the baseline dataframe
      :rtype: list[float]
    """

    model_wrapper = get_model_wrapper(task_type, target_column, baseline_dataframe)

    categorical_features = compute_categorical_features(baseline_dataframe, target_column)

    baseline_explanations = compute_explanations(model_wrapper, baseline_dataframe, categorical_features, target_column, task_type)
    _logger.info("Successfully computed explanations for baseline dataset")

    return baseline_explanations


@track(_get_logger)
def run(args):

    baseline_df = load_mltable_to_df(args.baseline_data)

    # todo: read from args when available
    task_type = "classification"
    target_column = "target"
    try:
        compute_feature_importance(task_type, target_column, baseline_df)
        _logger.info("Successfully executed the feature importance component.")
    except Exception as e:
        _logger.info("Error encountered when executing feature importance component: {0}", e)


if __name__ == "__main__":
    args = parse_args()

    run(args)
