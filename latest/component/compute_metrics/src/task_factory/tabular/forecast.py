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
        """
        Run the forecast.

        Args:
            X_test (_type_): _description_
            y_context (_type_): _description_
        """
        return self.model._model_impl.forecast(X_test, y_context, ignore_data_errors=True)

    def rolling_forecast(self, X_test, y_test, step=1):
        """
        Do the rolling forecast.

        Args:
            X_test (_type_): _description_
            y_test (_type_): _description_
            step (_type_): _description_
        """
        return self._model_impl.rolling_forecast(X_test, y_test, step=step, ignore_data_errors=True)
