# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Chat Completion."""

import pandas as pd
import numpy as np
from task_factory.base import PredictWrapper
from logging_utilities import get_logger
from constants import ChatCompletionConstants

logger = get_logger(name=__name__)


class ChatCompletion(PredictWrapper):
    """Chat Completion.

    Args:
        PredictWrapper (_type_): _description_
    """

    def predict(self, X_test, input_column_names, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)

        y_pred_combined = []  # list of dictionaries of predictions and predictions_appended
        y_pred_appended_convo = []  # prediction outputs appended to their entire conversation

        # loop 'cause model accepts a single dataframe (multi-turn conversation) as an input
        for _, X_df in X_test.iterrows():
            convo_arr = np.array(X_df[input_column_names[0]])
            if "generator_config" in kwargs:
                convo_arr_updated = {
                    "parameters": kwargs["generator_config"],
                    "inputs": {"input": convo_arr},
                }
            else:
                convo_arr_updated = convo_arr
            try:
                logger.info("Running predictions with the kwargs")
                y_pred = self.model.predict(convo_arr_updated, **kwargs)
            except TypeError:
                logger.info("Encountered TypeError,running predict ignoring the kwargs.")
                y_pred = self.model.predict(convo_arr_updated)
            except RuntimeError:
                logger.info("Encountered RuntimeError, retrying.")
                return self.handle_device_failure(convo_arr_updated, **kwargs)
            y_pred_template = [{'role': 'assistant', 'content': y_pred['output']}]
            y_pred_row = pd.DataFrame({input_column_names[0]: y_pred_template})
            y_pred_appended_convo = pd.concat([X_df, y_pred_row], ignore_index=True)
            y_pred_dict = {
                ChatCompletionConstants.OUTPUT: y_pred['output'],
                ChatCompletionConstants.OUTPUT_FULL_CONVERSATION: y_pred_appended_convo
            }
            y_pred_combined.append(y_pred_dict)

        return y_pred_combined
