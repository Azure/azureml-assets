# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utils functions for histogram computation."""
import math
import pyspark.sql.functions as F


def _get_smaller_df(baseline_df, production_df, baseline_count, production_count):
    """Get smaller dataframe."""
    return baseline_df if baseline_count < production_count else production_df


def _get_optimal_number_of_bins(baseline_df, production_df, baseline_count, production_count):
    """Calculate number of bins for histogram using struges alogorithm."""
    # TODO: Unnecessary calculation, use count from summary and remove _get_smaller_df()
    smaller_df = _get_smaller_df(
        baseline_df, production_df, baseline_count, production_count
    )
    num_bins = math.log2(smaller_df.count()) + 1
    return math.ceil(num_bins)


def get_dual_histogram_bin_edges(
    baseline_df, production_df, baseline_count, production_count, numerical_columns
):
    """Get histogram edges using fixed bin width."""
    num_bins = _get_optimal_number_of_bins(
        baseline_df, production_df, baseline_count, production_count
    )
    all_bin_edges = {}
    for col in numerical_columns:
        # TODO: profile agg if required
        baseline_col_min, baseline_col_max = baseline_df.agg(
            F.min(col), F.max(col)
        ).collect()[0]
        production_col_min, production_col_max = production_df.agg(
            F.min(col), F.max(col)
        ).collect()[0]

        min_value = min(baseline_col_min, production_col_min)
        max_value = max(baseline_col_max, production_col_max)

        bin_width = (max_value - min_value) / num_bins

        # If histogram has only one value then we only need a single bucket and
        # should skip the for-loop below.
        if min_value == max_value:
            delta = 0.005 if min_value == 0 else abs(min_value * 0.005)
            all_bin_edges[col] = [min_value - delta, min_value + delta]
            continue

        edges = []
        for bin in range(num_bins):
            bin = bin + 1
            if bin == 1:
                edges.append(min_value)
                edges.append(min_value + bin * bin_width)
            else:
                edges.append(min_value + bin * bin_width)

        all_bin_edges[col] = edges

    return all_bin_edges


def get_histograms(df, bin_edges, numerical_columns):
    """Compute bin edges that are suitable for both baseline_col and prod_col."""
    feature_histogram = {}

    for col in numerical_columns:
        if col not in bin_edges:
            continue
        hist_edge = bin_edges[col]
        # TODO: profile rdd.histogram
        histogram_col = df.select(col).rdd.flatMap(lambda x: x).histogram(hist_edge)
        feature_histogram[col] = histogram_col
    return feature_histogram
