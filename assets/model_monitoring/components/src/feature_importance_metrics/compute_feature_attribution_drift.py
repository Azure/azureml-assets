# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from shared_utilities.io_utils import read_mltable_in_spark, save_spark_df_as_mltable
from shared_utilities import constants

from feature_importance_metrics.feature_importance_utilities import convert_pandas_to_spark

from shared_utilities.patch_mltable import patch_all
patch_all()

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str)
    parser.add_argument("--production_data", type=str)
    parser.add_argument("--target_column", type=str)
    parser.add_argument("--signal_metrics", type=str)

    args = parser.parse_args()

    return args


def calculate_attribution_drift(baseline_explanations, production_explanations):
    """Compute feature attribution drift given two sets of explanations.

    :param explanations: feature importances with their corresponding feature names
    :type explanations: pandas.Dataframe
    :param production_explanations: list of explanations calculated using the production data
    :type production_explanations: list[float]
    :return: the ndcg metric between the baseline and production data
    :rtype: float
    """
    true_relevance = np.asarray([baseline_explanations[constants.METRIC_VALUE_COLUMN]])
    relevance_score = np.asarray([production_explanations[constants.METRIC_VALUE_COLUMN]])
    feature_attribution_drift = ndcg_score(true_relevance, relevance_score)
    _logger.info(f"feature attribution drift calculated: {feature_attribution_drift}")
    return feature_attribution_drift


def compute_ndcg_and_write_to_mltable(baseline_explanations, production_explanations,
                                      feature_attribution_data, baseline_row_count, production_row_count):
    """Write feature importance values to mltable.

    :param explanations: feature importances with their corresponding feature names
    :type explanations: pandas.Dataframe
    :param dataset: dataset to derive feature names
    :type dataset: pandas.Dataframe
    :param feature_attribution_data: path to folder to save mltable
    :type feature_attribution_data: string
    :param baseline_row_count: number of columns in baseline data
    :type baseline_row_count: number
    :param production_row_count: number of columns in production data
    :type production_row_count: number
    """
    metrics_data = pd.DataFrame(columns=[constants.FEATURE_NAME_COLUMN,
                                         constants.METRIC_VALUE_COLUMN,
                                         constants.FEATURE_CATEGORY_COLUMN,
                                         constants.METRIC_NAME_COLUMN,
                                         constants.THRESHOLD_VALUE])
    feature_attribution_drift = calculate_attribution_drift(baseline_explanations, production_explanations)

    ndcg_metric = {constants.FEATURE_NAME_COLUMN: "",
                   constants.METRIC_VALUE_COLUMN: feature_attribution_drift,
                   constants.METRIC_NAME_COLUMN: "NormalizedDiscountedCumulativeGain",
                   constants.FEATURE_CATEGORY_COLUMN: "",
                   constants.THRESHOLD_VALUE: float("nan")}
    metrics_data = metrics_data.append(ndcg_metric, ignore_index=True)
    baseline_row_count_data = {constants.FEATURE_NAME_COLUMN: "",
                               constants.METRIC_VALUE_COLUMN: baseline_row_count,
                               constants.METRIC_NAME_COLUMN: "BaselineRowCount",
                               constants.FEATURE_CATEGORY_COLUMN: "",
                               constants.THRESHOLD_VALUE: float("nan")}
    metrics_data = metrics_data.append(baseline_row_count_data, ignore_index=True)
    production_row_count_data = {constants.FEATURE_NAME_COLUMN: "",
                                 constants.METRIC_VALUE_COLUMN: production_row_count,
                                 constants.METRIC_NAME_COLUMN: "TargetRowCount",
                                 constants.FEATURE_CATEGORY_COLUMN: "",
                                 constants.THRESHOLD_VALUE: float("nan")}
    metrics_data = metrics_data.append(production_row_count_data, ignore_index=True)

    for (_, baseline_feature), (_, production_feature) in zip(baseline_explanations.iterrows(),
                                                              production_explanations.iterrows()):
        baseline_feature_importance_data = {
            constants.FEATURE_NAME_COLUMN: baseline_feature[constants.FEATURE_COLUMN],
            constants.METRIC_VALUE_COLUMN: baseline_feature[constants.METRIC_VALUE_COLUMN],
            constants.FEATURE_CATEGORY_COLUMN: baseline_feature[constants.FEATURE_CATEGORY_COLUMN],
            constants.METRIC_NAME_COLUMN: "BaselineFeatureImportance",
            constants.THRESHOLD_VALUE: float("nan")}
        production_feature_importance_data = {
            constants.FEATURE_NAME_COLUMN: production_feature[constants.FEATURE_COLUMN],
            constants.METRIC_VALUE_COLUMN: production_feature[constants.METRIC_VALUE_COLUMN],
            constants.FEATURE_CATEGORY_COLUMN: production_feature[constants.FEATURE_CATEGORY_COLUMN],
            constants.METRIC_NAME_COLUMN: "ProductionFeatureImportance",
            constants.THRESHOLD_VALUE: float("nan")}
        metrics_data = metrics_data.append(baseline_feature_importance_data, ignore_index=True)
        metrics_data = metrics_data.append(production_feature_importance_data, ignore_index=True)
    spark_data = convert_pandas_to_spark(metrics_data)
    save_spark_df_as_mltable(spark_data, feature_attribution_data)


def configure_data(data):
    """Convert pySpark.Dataframe to pandas.Dataframe and sort the data.

    :param data: feature importances with their corresponding feature names
    :type data: pySpark.Dataframe
    :return: the sorted pandas feature importances data with the row count dropped and the number of rows
    :rtype: tuple of pandas dataframe and number
    """
    df = read_mltable_in_spark(data).toPandas()
    for i in range(len(df.index)):
        if df.iloc[i][constants.METRIC_NAME_COLUMN] == constants.ROW_COUNT_COLUMN_NAME:
            num_rows = df.iloc[i][constants.METRIC_VALUE_COLUMN]
            df = df.drop(df.index[i])
    return [df.sort_values(by=[constants.FEATURE_COLUMN]), num_rows]


def drop_metadata_columns(baseline_data, production_data):
    """Drop any columns from production data that are not in the baseline data.

    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :param production_data: The production data meaning the most recent set of data
    sent to the model monitor, the current set of data
    :type production_data: pandas.DataFrame
    :return: production data with removed columns
    :rtype: pandas.DataFrame
    """
    baseline_data_features = baseline_data[constants.FEATURE_COLUMN].values
    production_data_features = production_data[constants.FEATURE_COLUMN].values
    for production_feature in production_data_features:
        if production_feature not in baseline_data_features:
            production_data = production_data.drop(
                production_data[production_data.feature == production_feature].index, axis=0)
            _logger.info(f"Dropped {production_feature} column in production dataset")
    return production_data


def run(args):
    """Calculate feature attribution drift."""
    try:
        [baseline_explanations, baseline_row_count] = configure_data(args.baseline_data)
        [production_explanations, production_row_count] = configure_data(args.production_data)
        production_explanations = drop_metadata_columns(baseline_explanations, production_explanations)
        compute_ndcg_and_write_to_mltable(baseline_explanations, production_explanations,
                                          args.signal_metrics, baseline_row_count, production_row_count)
        _logger.info("Successfully executed the feature attribution component.")
    except Exception as e:
        _logger.info(f"Error encountered when executing feature attribution component: {e}")
        raise e


if __name__ == "__main__":
    args = parse_args()

    run(args)
