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
            logger.warning("NER Doesn't support multi-gpu inference. Loading model to single GPU.")
            self._ensure_model_on_cpu()
            import torch
            if torch.cuda.is_available():
                kwargs["device"] = torch.cuda.current_device()
            else:
                kwargs["device"] = -1
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError:
            y_pred = self.model.predict(X_test)
        except RuntimeError as re:
            device = kwargs.get("device", -1)
            if device != -1:
                logger.info("Failed on GPU with error:" + repr(re))
                logger.warning("Predict failed on GPU. Falling back to CPU")
                self._ensure_model_on_cpu()
                kwargs["device"] = -1
                y_pred = self.model.predict(X_test, **kwargs)
            else:
                raise re

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[y_pred.columns[0]].values
        y_pred = list(map(lambda x: ast.literal_eval(x), y_pred))
        return y_pred
