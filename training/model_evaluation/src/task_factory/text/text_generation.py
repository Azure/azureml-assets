# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text Generation."""

from task_factory.base import PredictWrapper


class TextGenerator(PredictWrapper):
    """TextGenerator.

    Args:
        PredictWrapper (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError:
            y_pred = self.model.predict(X_test)

        return y_pred
