# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for data drift compute metrics component."""

import numpy as np
import pyspark.sql as pyspark_sql
import pyspark.sql.functions as F
from synapse.ml.exploratory import DistributionBalanceMeasure
from scipy.stats import chisquare
from io_utils import get_output_spark_df


def get_column_value_frequency(
    df: pyspark_sql.DataFrame, categorical_columns: list, total_count: int
):
    """Calculate frequency of value in a column."""
    entries = []
    for column in categorical_columns:
        new_df = (
            df.groupBy(column)
            .count()
            .withColumn("val_ratio", (F.col("count") / total_count))
        )
        entry = {str(row[column]): row["val_ratio"] for row in new_df.collect()}
        entries.append(entry)
    return entries


def _jensen_shannon_categorical(
    baseline_df: pyspark_sql.DataFrame,
    production_df: pyspark_sql.DataFrame,
    baseline_count: int,
    categorical_columns: list,
):
    """Jensen Shannon computation for categorical column."""
    reference_distribution = get_column_value_frequency(
        baseline_df, categorical_columns, baseline_count
    )

    output_df = (
        DistributionBalanceMeasure()
        .setSensitiveCols(categorical_columns)
        .setReferenceDistribution(reference_distribution)
        .transform(production_df.select(categorical_columns))
    )

    output_df = output_df.select("FeatureName", "DistributionBalanceMeasure.js_dist")
    output_df = (
        output_df.withColumnRenamed("FeatureName", "feature_name")
        .withColumnRenamed("js_dist", "metric_value")
        .withColumn("data_type", F.lit("Categorical"))
        .withColumn("metric_name", F.lit("JensenShannonDistance"))
    )

    return output_df


def _psi_categorical(
    baseline_df: pyspark_sql.DataFrame,
    production_df: pyspark_sql.DataFrame,
    baseline_count: int,
    production_count: int,
    categorical_columns: list,
):
    """PSI calculation for categorical columns."""
    # unique values observed in baseline and production dataset
    psi_rows = []
    for column in categorical_columns:
        print("psi_column: ", column)
        baseline_prod_unique_values = (
            production_df.select(column).union(baseline_df.select(column)).distinct()
        )
        all_unique_values = baseline_prod_unique_values.rdd.flatMap(
            lambda x: x
        ).collect()
        all_unique_values_dict = {str(value): 0 for value in all_unique_values}

        count_baseline_df = baseline_df.groupBy(column).count()
        count_baseline_df_dict = {
            str(row[column]): row["count"] for row in count_baseline_df.collect()
        }
        baseline_frequencies = {**all_unique_values_dict, **count_baseline_df_dict}

        baseline_ratios = {}
        for value in baseline_frequencies:
            baseline_ratios[value] = (baseline_frequencies[value] + 1) / baseline_count

        count_prod_df = production_df.groupBy(column).count()
        count_prod_df_dict = {
            str(row[column]): row["count"] for row in count_prod_df.collect()
        }
        production_frequencies = {**all_unique_values_dict, **count_prod_df_dict}

        production_ratios = {}
        for value in production_frequencies:
            # Add 1 to every value to avoid infinity when value is 0
            production_ratios[value] = (
                production_frequencies[value] + 1
            ) / production_count

        psi = 0
        for i in baseline_ratios:
            psi += (production_ratios[i] - baseline_ratios[i]) * np.log(
                production_ratios[i] / baseline_ratios[i]
            )

        psi_row = [column, float(psi), "categorical", "PopulationStabilityIndex"]
        psi_rows.append(psi_row)

    output_df = get_output_spark_df(psi_rows)
    return output_df


def _chisquaretest(
    baseline_df, production_df, baseline_count, production_count, categorical_columns
) -> float:
    """Test to determine data drift using Pearson's Chisquare Test for each feature."""
    chisqtest_rows = []

    for column in categorical_columns:
        print(column)
        baseline_prod_unique_values = (
            production_df.select(column).union(baseline_df.select(column)).distinct()
        )
        all_unique_values = baseline_prod_unique_values.rdd.flatMap(
            lambda x: x
        ).collect()
        initial_unique_values_dict = {str(value): 0 for value in all_unique_values}

        count_baseline_df = baseline_df.groupBy(column).count()
        count_baseline_df_dict = {
            str(row[column]): row["count"] for row in count_baseline_df.collect()
        }

        baseline_frequencies = {**initial_unique_values_dict, **count_baseline_df_dict}

        baseline_ratios = {}
        for value in baseline_frequencies:
            baseline_ratios[value] = (baseline_frequencies[value]) / baseline_count

        count_prod_df = production_df.groupBy(column).count()
        count_prod_df_dict = {
            str(row[column]): row["count"] for row in count_prod_df.collect()
        }

        production_frequencies_map = {
            **initial_unique_values_dict,
            **count_prod_df_dict,
        }
        production_frequencies = list(production_frequencies_map.values())

        for key in baseline_ratios:
            baseline_ratios[key] *= production_count
        expected_frequency = list(baseline_ratios.values())

        chisqtest_val = chisquare(production_frequencies, expected_frequency).pvalue
        chisqtest_row = [
            column,
            float(chisqtest_val),
            "categorical",
            "PearsonsChiSquaredTest",
        ]
        chisqtest_rows.append(chisqtest_row)

    output_df = get_output_spark_df(chisqtest_rows)
    return output_df


def compute_categorical_data_drift_measures_tests(
    baseline_df: pyspark_sql.DataFrame,
    production_df: pyspark_sql.DataFrame,
    baseline_count: int,
    production_count: int,
    categorical_metric: str,
    categorical_columns: list,
    categorical_threshold: float,
):
    """Compute Data drift metrics and tests for numerical columns."""
    if categorical_metric == "JensenShannonDistance":
        output_df = _jensen_shannon_categorical(
            baseline_df, production_df, baseline_count, categorical_columns
        )
    elif categorical_metric == "PopulationStabilityIndex":
        output_df = _psi_categorical(
            baseline_df,
            production_df,
            baseline_count,
            production_count,
            categorical_columns,
        )
    elif categorical_metric == "PearsonsChiSquaredTest":
        output_df = _chisquaretest(
            baseline_df,
            production_df,
            baseline_count,
            production_count,
            categorical_columns,
        )
    else:
        raise Exception(f"Invalid metric {categorical_metric} for categorical feature")

    output_df = output_df.withColumn(
        "threshold_value", F.lit(categorical_threshold).cast("float")
    )
    return output_df
