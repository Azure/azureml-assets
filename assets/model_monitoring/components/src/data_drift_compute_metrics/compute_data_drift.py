# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift compute metrics component."""


import pyspark.sql as pyspark_sql
import pyspark.sql.functions as F
from numerical_data_drift_metrics import compute_numerical_data_drift_measures_tests
from categorical_data_drift_metrics import compute_categorical_data_drift_measures_tests
from io_utils import get_output_spark_df
from shared_utilities.df_utils import (
    get_common_columns,
    get_feature_type_override_map,
    get_numerical_cols_with_df_with_override,
    get_categorical_cols_with_df_with_override
)


def compute_data_drift_measures_tests(
    baseline_df: pyspark_sql.DataFrame,
    production_df: pyspark_sql.DataFrame,
    numerical_metric: str,
    categorical_metric: str,
    numerical_threshold: str,
    categorical_threshold: str,
    override_numerical_features: str,
    override_categorical_features: str
):
    """Compute Data drift metrics and tests."""
    common_columns_dict = get_common_columns(baseline_df, production_df)
    feature_type_override_map = get_feature_type_override_map(override_numerical_features, override_categorical_features)
    print("Getting feature type override map: ", feature_type_override_map)

    numerical_columns_names = get_numerical_cols_with_df_with_override(feature_type_override_map, 
                                                                       baseline_df,
                                                                       common_columns_dict)
    categorical_columns_names = get_categorical_cols_with_df_with_override(feature_type_override_map, 
                                                                       baseline_df,
                                                                       common_columns_dict)
    baseline_df = baseline_df.dropna()
    production_df = production_df.dropna()

    baseline_df_count = baseline_df.count()
    production_df_count = production_df.count()

    numerical_baseline_df = baseline_df.select(numerical_columns_names)
    categorical_baseline_df = baseline_df.select(categorical_columns_names)

    numerical_production_df = production_df.select(numerical_columns_names)
    categorical_production_df = production_df.select(categorical_columns_names)

    if len(numerical_columns_names) == 0 and \
       len(categorical_columns_names) == 0:
        raise ValueError("No common columns found between production data and baseline"
              "data. We dont support this scenario.")

    if len(numerical_columns_names) != 0:
        numerical_df = compute_numerical_data_drift_measures_tests(
            numerical_baseline_df,
            numerical_production_df,
            baseline_df_count,
            production_df_count,
            numerical_metric,
            numerical_columns_names,
            numerical_threshold,
        )

    if len(categorical_columns_names) != 0:
        categorical_df = compute_categorical_data_drift_measures_tests(
            categorical_baseline_df,
            categorical_production_df,
            baseline_df_count,
            production_df_count,
            categorical_metric,
            categorical_columns_names,
            categorical_threshold,
        )
    # TODO: fix this if, else
    if len(numerical_columns_names) != 0 and len(categorical_columns_names) != 0:
        output_df = numerical_df.union(categorical_df)
    elif len(numerical_columns_names) != 0:
        output_df = numerical_df
    else:
        output_df = categorical_df

    baseline_count_row = [
        "",
        float(baseline_df_count),
        "",
        "BaselineRowCount",
        "",
        ""
    ]
    target_count_row = [
        "",
        float(production_df_count),
        "",
        "TargetRowCount",
        "",
        ""
    ]
    row_count_metric_df = get_output_spark_df([baseline_count_row,
                                               target_count_row])
    row_count_metric_df = row_count_metric_df \
        .withColumn("threshold_value", F.lit("nan").cast("float"))
    output_df = output_df.union(row_count_metric_df)

    return output_df
