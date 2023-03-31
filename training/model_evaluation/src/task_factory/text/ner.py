"""Text NER."""

from task_factory.base import PredictWrapper
import pandas as pd
import ast


class TextNerPredictor(PredictWrapper):
    """TextNER Predictor.

    Args:
        PredictWrapper (_type_): _description_
    """

    def predict(self, X_test, **kwargs):
        """Get Entity labels.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError:
            y_pred = self.model.predict(X_test)

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[y_pred.columns[0]].values
        y_pred = list(map(lambda x: ast.literal_eval(x), y_pred))
        return y_pred
