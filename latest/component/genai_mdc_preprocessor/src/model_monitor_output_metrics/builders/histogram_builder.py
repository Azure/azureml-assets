# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates histograms."""

from typing import List
from pyspark.sql import Row


class HistogramBuilder:
    """Builder class which creates a histogram file."""

    def __init__(self, target_histograms: List[Row], baseline_histograms: List[Row]):
        """Construct a HistogramBuilder instance."""
        histograms = {}

        if target_histograms is None and baseline_histograms is None:
            self.histograms = {}
            return

        if baseline_histograms:
            for row in baseline_histograms:
                feature_name = row["feature_bucket"]
                if feature_name not in histograms:
                    histograms[feature_name] = {
                        "featureName": row["feature_bucket"],
                        "histogram": {},
                    }

                bucket = {
                    "baselineCount": row["bucket_count"],
                }

                bucket_key = None
                if row["data_type"] == "categorical":
                    bucket_key = row["category_bucket"]
                    bucket["category"] = row["category_bucket"]
                else:
                    bucket_key = row["lower_bound"]
                    bucket["lowerBound"] = row["lower_bound"]
                    bucket["upperBound"] = row["upper_bound"]

                histograms[feature_name]["histogram"][bucket_key] = bucket

        if target_histograms:
            for row in target_histograms:
                feature_name = row["feature_bucket"]
                if feature_name not in histograms:
                    continue
                if row["data_type"] == "categorical":
                    if row["category_bucket"] in histograms[feature_name]["histogram"]:
                        histograms[feature_name]["histogram"][row["category_bucket"]][
                            "targetCount"
                        ] = row["bucket_count"]
                else:
                    histograms[feature_name]["histogram"][row["lower_bound"]][
                        "targetCount"
                    ] = row["bucket_count"]

        self.histograms = {}
        for feature_name in histograms.keys():
            self.histograms[feature_name] = {
                "featureName": feature_name,
                "histogram": list(histograms[feature_name]["histogram"].values()),
            }

    def get_features(self) -> List[str]:
        """Get feature names."""
        return self.histograms.keys()

    def build(self, feature_name: str):
        """Build histogram file content."""
        if feature_name not in self.histograms:
            return None
        return self.histograms[feature_name]
