# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text Classifier."""

from task_factory.tabular.classification import TabularClassifier


class TextClassifier(TabularClassifier):
    """Text Classifier.

    Args:
        TabularClassifier (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Predict labels.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        return super().predict(X_test, **kwargs)

    def predict_proba(self, X_test, **kwargs):
        """Get prediction probabilities.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        return super().predict_proba(X_test, **kwargs)
