# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fill Mask."""

from task_factory.base import PredictWrapper
from logging_utilities import get_logger

logger = get_logger(name=__name__)


class FillMask(PredictWrapper):
    """FillMask.

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
        except RuntimeError:
            return self.handle_device_failure(X_test, **kwargs)

        return y_pred
