# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Chat Completion."""

import pandas as pd
from task_factory.base import PredictWrapper
from logging_utilities import get_logger

logger = get_logger(name=__name__)


class ChatCompletion(PredictWrapper):
    """Chat Completion.

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

        y_pred_combined = []
        for X_df in X_test:
            try:
                y_pred = self.model.predict(X_df, **kwargs)
            except TypeError:
                logger.info("Encountered TypeError,running predict ignorning the kwargs.")
                y_pred = self.model.predict(X_df)
            except RuntimeError:
                logger.info("Encountered RuntimeError, retrying.")
                return self.handle_device_failure(X_df, **kwargs)
            y_pred_template = [{'role': 'assistant', 'content': y_pred['output']}]
            y_pred_row = pd.DataFrame({"input_string": y_pred_template})
            X_df = pd.concat([X_df, y_pred_row], ignore_index=True)

            y_pred_combined.append(X_df)

        return y_pred_combined
