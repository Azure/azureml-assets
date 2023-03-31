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
        return super().predict(X_test, **kwargs)
