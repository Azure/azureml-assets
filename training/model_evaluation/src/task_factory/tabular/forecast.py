# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Forecast Predictor."""

# TODO: Add Forecast predictor

from task_factory.base import ForecastWrapper


class TabularForecast(ForecastWrapper):
    """Forecaster.

    Args:
        ForecastWrapper (_type_): _description_
    """

    def forecast(self, X_test, y_context, **kwargs):
        """TODO: Add forecast."""
        ...
