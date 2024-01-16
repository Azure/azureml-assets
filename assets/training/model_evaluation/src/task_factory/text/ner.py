# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text NER."""

from task_factory.base import PredictWrapper
import pandas as pd
import ast
from logging_utilities import get_logger

logger = get_logger(name=__name__)


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
        self._ensure_base_model_input_schema(X_test)
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError:
            y_pred = self.model.predict(X_test)
        except RuntimeError:
            return self.handle_device_failure(X_test, **kwargs)

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[y_pred.columns[0]].values
        y_pred = list(map(lambda x: ast.literal_eval(x), y_pred))
        return y_pred
