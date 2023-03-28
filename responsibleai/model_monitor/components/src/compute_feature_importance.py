# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import logging
import pandas as pd

from feature_importance_utilities import get_model_wrapper, compute_explanations,  compute_categorical_features
from io_utils import load_mltable_to_df, save_df_as_mltable

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
    parser.add_argument("--feature_importance_data", type=str)

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
    _logger.info("Successfully computed explanations for dataset")

    return baseline_explanations


def write_to_mltable(explanations, dataset, file_path):
    """write feature importance values to mltable
      :param explanations: list of feature importances in the order of the baseline columns
      :type explanations: list[float]
      :param dataset: dataset to derive feature names
      :type dataset: pandas.Dataframe
      :param file_path: path to folder to save mltable
      :type file_path: string
    """
    metrics_dataframe = pd.DataFrame(columns=['feature', 'metric_value', 'metric_name'])
    for index in range(len(explanations)):
        new_row = {"feature": dataset.columns, "metic_value": explanations[index], "metric_name": "feature_importance"}
        metrics_dataframe = metrics_dataframe.append(new_row, ignore_index=True)
    save_df_as_mltable(metrics_dataframe, file_path)


@track(_get_logger)
def run(args):

    baseline_df = load_mltable_to_df(args.baseline_data)

    # todo: read from args when available
    task_type = "classification"
    target_column = "target"
    try:
        feature_importances = compute_feature_importance(task_type, target_column, baseline_df)
        _logger.info("Successfully executed the feature importance component.")
        write_to_mltable(feature_importances, baseline_df, args.feature_importance_data)
    except Exception as e:
        _logger.info("Error encountered when executing feature importance component: {0}", e)


if __name__ == "__main__":
    args = parse_args()

    run(args)
