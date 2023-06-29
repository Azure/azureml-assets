# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression Predictor."""

from task_factory.base import PredictWrapper


class TabularRegressor(PredictWrapper):
    """Tabular Regressor.

    Args:
        PredictWrapper (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Regression inference.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.is_torch:
            y_pred = self.model.predict(X_test, device=kwargs.get("device", -1))
        else:
            y_pred = self.model.predict(X_test)
        return y_pred
