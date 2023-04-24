# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text Regression."""

from task_factory.tabular.regression import TabularRegressor


class TextRegressor(TabularRegressor):
    """Text Regressor.

    Args:
        TabularRegressor (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        return super().predict(X_test, **kwargs)
