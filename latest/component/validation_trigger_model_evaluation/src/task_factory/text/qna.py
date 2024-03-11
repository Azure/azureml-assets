# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Question Answering."""

from task_factory.base import PredictWrapper
from logging_utilities import get_logger

logger = get_logger(name=__name__)


class QnAPredictor(PredictWrapper):
    """QnA Predictor.

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
        self._ensure_base_model_input_schema(X_test)
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError as e:
            logger.warning(f"TypeError exception raised. Reason: {e}")
            y_pred = self.model.predict(X_test)
        except RuntimeError as e:
            logger.warning(f"RuntimeError exception raised. Reason: {e}")
            return self.handle_device_failure(X_test, **kwargs)

        return y_pred
