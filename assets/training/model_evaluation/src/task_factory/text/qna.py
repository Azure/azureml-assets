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
        except TypeError:
            y_pred = self.model.predict(X_test)
        except RuntimeError as re:
            device = kwargs.get("device", -1)
            if device != -1:
                logger.warning("Predict failed on GPU. Falling back to CPU")
                self._ensure_model_on_cpu()
                kwargs["device"] = -1
                y_pred = self.model.predict(X_test, **kwargs)
            else:
                raise re

        return y_pred
