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

        # loop 'cause model accepts a single dataframe (multi-turn conversation) as an input
        for idx, X_df in X_test.iterrows():
            logger.info(f"Predicting for chat item: {idx}")
            convo_df = pd.DataFrame(np.array(X_df[input_column_names[0]]), columns=[input_column_names[0]])
            if convo_df[input_column_names[0]].iloc[-1]['role'] == 'assistant':
                convo_df = convo_df[:-1]
            if "generator_config" in kwargs:
                convo_arr_updated = {
                    "parameters": kwargs["generator_config"],
                    "inputs": {"input": convo_df},
                }
            else:
                convo_arr_updated = convo_df
            try:
                logger.info("Running predictions with the kwargs")
                y_pred = self.model.predict(convo_arr_updated, **kwargs)
            except TypeError as e:
                logger.warning(f"Encountered TypeError.  Reason: {e}")
                logger.info("Running predict ignoring the kwargs.")
                y_pred = self.model.predict(convo_arr_updated)
            except RuntimeError as e:
                logger.warning(f"Encountered RuntimeError.  Reason: {e}")
                logger.info("Retrying.")
                return self.handle_device_failure(convo_arr_updated, **kwargs)
            if isinstance(y_pred, pd.DataFrame):
                y_pred = y_pred[y_pred.columns[0]].tolist()[0]
            elif isinstance(y_pred, dict):
                y_pred = y_pred["output"]
            elif isinstance(y_pred, list) and len(y_pred) == 1:
                y_pred = y_pred[0]

            y_pred_template = pd.DataFrame(
                np.array([{'role': 'assistant', 'content': y_pred}]), columns=[input_column_names[0]]
            )

            # prediction outputs appended to their entire conversation
            y_pred_appended_convo = pd.concat([convo_df, y_pred_template], ignore_index=True)

            y_pred_dict = {
                ChatCompletionConstants.OUTPUT: y_pred,
                ChatCompletionConstants.OUTPUT_FULL_CONVERSATION: [{
                    input_column_names[0]: np.array(y_pred_appended_convo[input_column_names[0]].tolist())
                }]
            }
            y_pred_combined.append(y_pred_dict)

        return y_pred_combined
