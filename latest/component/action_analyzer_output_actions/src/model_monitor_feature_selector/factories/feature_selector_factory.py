# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates Feature Selectors."""
import pyspark.sql as pyspark_sql

from model_monitor_feature_selector.selectors.feature_selector_top_by_attribution import (
    FeatureSelectorTopNByAttribution,
)
from model_monitor_feature_selector.selectors.feature_selector_type import (
    FeatureSelectorType,
)
from model_monitor_feature_selector.selectors.feature_selector import FeatureSelector
from model_monitor_feature_selector.selectors.feature_selector_all import (
    FeatureSelectorAll,
)
from model_monitor_feature_selector.selectors.feature_selector_subset import (
    FeatureSelectorSubset,
)


class FeatureSelectorFactory:
    """Builder class which creates a Signal output."""

    def produce(
        self,
        feature_selector_type: str,
        filter_value: str,
        feature_importance: pyspark_sql.DataFrame,
    ) -> FeatureSelector:
        """Produce a signal of the given type."""
        if feature_selector_type == FeatureSelectorType.ALL.name:
            return FeatureSelectorAll()
        elif feature_selector_type == FeatureSelectorType.SUBSET.name:
            feature_names = filter_value.split(",")
            return FeatureSelectorSubset(feature_names=feature_names)
        elif feature_selector_type == FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name:
            if not filter_value.isdigit():
                raise ValueError(
                    "Invalid feature value. Please provide a valid integer value"
                    + f" when leveraging '{FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name}'."
                )
            return FeatureSelectorTopNByAttribution(
                filter_value=int(filter_value), feature_importance=feature_importance
            )
        else:
            raise Exception(
                f"Invalid feature selector type '{feature_selector_type}'. Available feature selectors are [{FeatureSelectorType.ALL.name}, {FeatureSelectorType.SUBSET.name}, {FeatureSelectorType.TOP_N_BY_ATTRIBUTION.name}]" # noqa
            )
