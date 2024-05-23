# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score

from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_momo_component_environment,
)
from shared_utilities import constants
from shared_utilities.momo_exceptions import InvalidInputError

from feature_importance_metrics.feature_importance_utilities import convert_pandas_to_spark, log_time_and_message


CORRELATION_ID = 'correlationid'
COLUMN_MISMATCH_ERR = "Column names in baseline and production data do not match, diff: {diff}"
FEATURE_MISMATCH_ERR = "Feature names in baseline and production data do not match, diff: {diff}"


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
    log_time_and_message(f"feature attribution drift calculated: {feature_attribution_drift}")
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
    feature_attribution_drift = calculate_attribution_drift(baseline_explanations, production_explanations)
    data = []
    log_time_and_message("Begin writing metric to mltable")
    ndcg_metric = pd.DataFrame({constants.FEATURE_NAME_COLUMN: "",
                                constants.METRIC_VALUE_COLUMN: feature_attribution_drift,
                                constants.METRIC_NAME_COLUMN: "NormalizedDiscountedCumulativeGain",
                                constants.FEATURE_CATEGORY_COLUMN: "",
                                constants.GROUP_COLUMN: "",
                                constants.GROUP_PIVOT_COLUMN: "",
                                constants.THRESHOLD_VALUE: float("nan")}, index=[0])
    data.append(ndcg_metric)
    baseline_row_count_data = pd.DataFrame({constants.FEATURE_NAME_COLUMN: "",
                                            constants.METRIC_VALUE_COLUMN: baseline_row_count,
                                            constants.METRIC_NAME_COLUMN: "BaselineRowCount",
                                            constants.FEATURE_CATEGORY_COLUMN: "",
                                            constants.GROUP_COLUMN: "",
                                            constants.GROUP_PIVOT_COLUMN: "",
                                            constants.THRESHOLD_VALUE: float("nan")}, index=[0])
    data.append(baseline_row_count_data)
    production_row_count_data = pd.DataFrame({constants.FEATURE_NAME_COLUMN: "",
                                              constants.METRIC_VALUE_COLUMN: production_row_count,
                                              constants.METRIC_NAME_COLUMN: "TargetRowCount",
                                              constants.FEATURE_CATEGORY_COLUMN: "",
                                              constants.GROUP_COLUMN: "",
                                              constants.GROUP_PIVOT_COLUMN: "",
                                              constants.THRESHOLD_VALUE: float("nan")}, index=[0])
    data.append(production_row_count_data)

    for (_, baseline_feature), (_, production_feature) in zip(baseline_explanations.iterrows(),
                                                              production_explanations.iterrows()):
        baseline_feature_importance_data = pd.DataFrame({
                    constants.FEATURE_NAME_COLUMN: baseline_feature[constants.FEATURE_COLUMN],
                    constants.METRIC_VALUE_COLUMN: baseline_feature[constants.METRIC_VALUE_COLUMN],
                    constants.FEATURE_CATEGORY_COLUMN: baseline_feature[constants.FEATURE_CATEGORY_COLUMN],
                    constants.METRIC_NAME_COLUMN: "BaselineFeatureImportance",
                    constants.GROUP_COLUMN: baseline_feature[constants.FEATURE_CATEGORY_COLUMN],
                    constants.GROUP_PIVOT_COLUMN: "",
                    constants.THRESHOLD_VALUE: float("nan")}, index=[0])
        production_feature_importance_data = pd.DataFrame({
            constants.FEATURE_NAME_COLUMN: production_feature[constants.FEATURE_COLUMN],
            constants.METRIC_VALUE_COLUMN: production_feature[constants.METRIC_VALUE_COLUMN],
            constants.FEATURE_CATEGORY_COLUMN: production_feature[constants.FEATURE_CATEGORY_COLUMN],
            constants.METRIC_NAME_COLUMN: "ProductionFeatureImportance",
            constants.GROUP_COLUMN: baseline_feature[constants.FEATURE_CATEGORY_COLUMN],
            constants.GROUP_PIVOT_COLUMN: "",
            constants.THRESHOLD_VALUE: float("nan")}, index=[0])
        data.append(baseline_feature_importance_data)
        data.append(production_feature_importance_data)

    metrics_data = pd.DataFrame(
            columns=[constants.FEATURE_NAME_COLUMN,
                     constants.METRIC_VALUE_COLUMN,
                     constants.FEATURE_CATEGORY_COLUMN,
                     constants.METRIC_NAME_COLUMN,
                     constants.GROUP_COLUMN,
                     constants.GROUP_PIVOT_COLUMN,
                     constants.THRESHOLD_VALUE])
    metrics_data = pd.concat(data)
    spark_data = convert_pandas_to_spark(metrics_data)
    save_spark_df_as_mltable(spark_data, feature_attribution_data)


def configure_data(data):
    """Convert pySpark.Dataframe to pandas.Dataframe and sort the data.

    :param data: feature importances with their corresponding feature names
    :type data: pySpark.Dataframe
    :return: the sorted pandas feature importances data with the row count dropped and the number of rows
    :rtype: tuple of pandas dataframe and number
    """
    df = data.toPandas()
    for i in range(len(df.index)):
        if df.iloc[i][constants.METRIC_NAME_COLUMN] == constants.ROW_COUNT_COLUMN_NAME:
            num_rows = df.iloc[i][constants.METRIC_VALUE_COLUMN]
            df = df.drop(df.index[i])
    return [df.sort_values(by=[constants.FEATURE_COLUMN]), num_rows]


def drop_correlation_id_row(df):
    """Drop the row with the correlationid feature name."""
    df.drop(df.loc[df[constants.FEATURE_COLUMN] == CORRELATION_ID].index, inplace=True)


def check_column_diff(baseline, production, error_message):
    """Check if the columns in the baseline and production data match."""
    if baseline != production:
        diff = baseline ^ production
        error = error_message.format(diff=diff)
        log_time_and_message(error)
        raise InvalidInputError(error)


def get_feature_columns(df):
    """Get the feature columns from the dataframe."""
    return set(df[constants.FEATURE_COLUMN].to_list())


def validate_columns_match(baseline_df, production_df):
    """Validate that the columns in the baseline and production data match."""
    baseline_columns = set(baseline_df.columns)
    production_columns = set(production_df.columns)
    check_column_diff(baseline_columns, production_columns, COLUMN_MISMATCH_ERR)
    validate_features_match(baseline_df, production_df)


def validate_features_match(baseline_df, production_df):
    """Validate that the feature columns in the baseline and production data match."""
    baseline_features = get_feature_columns(baseline_df)
    production_features = get_feature_columns(production_df)
    # drop correlationid column from both base and prod data
    if CORRELATION_ID in baseline_features:
        drop_correlation_id_row(baseline_df)
        baseline_features = get_feature_columns(baseline_df)
    if CORRELATION_ID in production_features:
        drop_correlation_id_row(production_df)
        production_features = get_feature_columns(production_df)
    check_column_diff(baseline_features, production_features, FEATURE_MISMATCH_ERR)


def run(args):
    """Calculate feature attribution drift."""
    # setup momo environment
    init_momo_component_environment()

    try:
        log_time_and_message("Reading in baseline data & target data")
        baseline_df = try_read_mltable_in_spark_with_error(args.baseline_data, "baseline_data")
        production_df = try_read_mltable_in_spark_with_error(args.production_data, "production_data")
        [baseline_explanations, baseline_row_count] = configure_data(baseline_df)
        [production_explanations, production_row_count] = configure_data(production_df)
        validate_columns_match(baseline_explanations, production_explanations)
        compute_ndcg_and_write_to_mltable(baseline_explanations, production_explanations,
                                          args.signal_metrics, baseline_row_count, production_row_count)
        log_time_and_message("Successfully executed the feature attribution component.")
    except Exception as e:
        log_time_and_message(f"Error encountered when executing feature attribution component: {e}")
        raise e


if __name__ == "__main__":
    args = parse_args()

    run(args)
