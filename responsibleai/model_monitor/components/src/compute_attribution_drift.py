# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import ndcg_score
from azureml.exceptions import UserErrorException
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
    parser.add_argument("--production_data", type=str)
    parser.add_argument("--feature_attribution_data", type=str)

    args = parser.parse_args()

    return args


def calculate_attribution_drift(baseline_explanations, production_explanations):
    """Compute feature attribution drift given two sets of explanations

      :param baseline_explanations: list of explanations calculated using the baseline dataframe
      :type baseline_explanations: list[float]
      :param production_explanations: list of explanations calculated using the production dataframe
      :type production_explanations: list[float]
      :return: the ndcg metric between the baseline and production data
      :rtype: float
    """
    true_relevance = np.asarray([baseline_explanations])
    relevance_score = np.asarray([production_explanations])
    feature_attribution_drift = ndcg_score(true_relevance, relevance_score)
    # just log for now, eventually we will have to write the output
    _logger.info("feature attribution drift calculated: {0}", feature_attribution_drift)
    return feature_attribution_drift


def compute_attribution_drift(task_type, target_column, baseline_dataframe, production_dataframe, feature_attribution_data):
    """Compute feature attribution drift by calculating feature importances on each
    dataframe input and using these to calculate the ndcg metric

      :param task_type: The task type (regression or classification) of the resulting model
      :type task_type: string
      :param target_column: the column to predict
      :type target_column: string
      :param baseline_dataframe: The baseline data meaning the data used to create the
      model monitor
      :type baseline_dataframe: pandas.DataFrame
      :param production_dataframe: The production data meaning the most recent set of data
      sent to the model monitor, the current set of data
      :type production_dataframe: pandas.DataFrame
      :return: the ndcg metric between the baseline and production data
      :rtype: float
    """
    if len(baseline_dataframe.columns.difference(production_dataframe.columns)) > 0:
        raise UserErrorException("Dataset columns differ in baseline and production datasets")

    model_wrapper = get_model_wrapper(task_type, target_column, baseline_dataframe, production_dataframe)

    categorical_features = compute_categorical_features(baseline_dataframe, target_column)

    baseline_explanations = compute_explanations(model_wrapper, baseline_dataframe, categorical_features, target_column, task_type)
    _logger.info("Successfully computed explanations for baseline dataset")

    production_explanations = compute_explanations(model_wrapper, production_dataframe, categorical_features, target_column, task_type)
    _logger.info("Successfully computed explanations for production dataset")

    write_to_mltable(baseline_explanations, production_explanations, feature_attribution_data)


def write_to_mltable(baseline_explanations, production_explanations, feature_attribution_data):
    """write feature importance values to mltable
      :param explanations: list of feature importances in the order of the baseline columns
      :type explanations: list[float]
      :param dataset: dataset to derive feature names
      :type dataset: pandas.Dataframe
      :param file_path: path to folder to save mltable
      :type file_path: string
    """
    metrics_dataframe = pd.DataFrame(columns=['metric_value', 'metric_name'])
    feature_attribution_drift = calculate_attribution_drift(baseline_explanations, production_explanations)
    ndcg_metric = {'metric_value': feature_attribution_drift, 'metric_name': "normalized_discounted_cumulative_gain"}
    metrics_dataframe = metrics_dataframe.append(ndcg_metric, ignore_index=True)
    save_df_as_mltable(metrics_dataframe, feature_attribution_data)

@track(_get_logger)
def run(args):

    baseline_df = load_mltable_to_df(args.baseline_data)
    production_df = load_mltable_to_df(args.production_data)

    # todo: read from args when available
    task_type = "classification"
    target_column = "target"
    try:
        compute_attribution_drift(task_type, target_column, baseline_df, production_df, args.feature_attribution_data)
        _logger.info("Successfully executed the feature attribution component.")
    except Exception as e:
        _logger.info("Error encountered when executing feature attribution component: {0}", e)


if __name__ == "__main__":
    args = parse_args()

    run(args)
