# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the histogram builder class."""

import pytest
from typing import List
from pyspark.sql import Row
from model_monitor_output_metrics.builders.histogram_builder import HistogramBuilder


@pytest.mark.unit1
class TestHistogramBuilder:
    """Test class for data drift output metrics component."""

    def test_numerical_histogram_builder(self):
        """Test generate meta file content for success scenario."""
        feature_name = "petal_length"
        baseline_histogram: List[Row] = [
            Row(
                feature_bucket=feature_name,
                bucket_count=10,
                lower_bound=1.0,
                upper_bound=2.0,
                data_type="numerical",
            ),
            Row(
                feature_bucket=feature_name,
                bucket_count=20,
                lower_bound=2.0,
                upper_bound=3.0,
                data_type="numerical",
            ),
        ]

        target_histogram: List[Row] = [
            Row(
                feature_bucket=feature_name,
                bucket_count=5,
                lower_bound=1.0,
                upper_bound=2.0,
                data_type="numerical",
            ),
            Row(
                feature_bucket=feature_name,
                bucket_count=7,
                lower_bound=2.0,
                upper_bound=3.0,
                data_type="numerical",
            ),
        ]

        target = HistogramBuilder(
            target_histograms=target_histogram, baseline_histograms=baseline_histogram
        ).build(feature_name)

        assert target["featureName"] == feature_name
        assert len(target["histogram"]) == 2

        for i in range(0, 2):
            assert (
                target["histogram"][i]["baselineCount"]
                == baseline_histogram[i]["bucket_count"]
            )
            assert (
                target["histogram"][i]["targetCount"]
                == target_histogram[i]["bucket_count"]
            )
            assert "category" not in target["histogram"][i]
            assert (
                target["histogram"][i]["lowerBound"]
                == baseline_histogram[i]["lower_bound"]
            )
            assert (
                target["histogram"][i]["upperBound"]
                == baseline_histogram[i]["upper_bound"]
            )
