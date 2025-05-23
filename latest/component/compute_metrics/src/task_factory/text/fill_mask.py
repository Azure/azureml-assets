# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Fill Mask."""
import pandas as pd
from task_factory.base import PredictWrapper
from logging_utilities import get_logger
from constants import MODEL_FLAVOR

logger = get_logger(name=__name__)


class FillMask(PredictWrapper):
    """FillMask.

    Args:
        PredictWrapper (_type_): _description_
    """

    # This method should not be invoked here. The predictions should ideally be forwarded to compute_metrics
    # which can then take care of further operation
    def get_tokenizer_mask(self):
        """Get tokenizer mask."""
        try:
            if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
                return self.model._model_impl.pipeline.tokenizer.mask_token
            elif self.model_flavor == MODEL_FLAVOR.HFTRANSFORMERSV2 or \
                    self.model_flavor == MODEL_FLAVOR.HFTRANSFORMERS:
                return self.model._model_impl.tokenizer.mask_token
            else:
                return None
        except Exception as e:
            logger.error(f"Error while fetching tokenizer mask: {e}")
            return None

    def return_masked_text(self, X_test, y_pred):
        """Return masked text."""
        tokenizer_mask = self.get_tokenizer_mask()
        indexes = y_pred.index if isinstance(y_pred, pd.DataFrame) else None
        if tokenizer_mask is None:
            return y_pred
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[y_pred.columns[0]].to_list()
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test[X_test.columns[0]].to_list()
        # Any other type of input is not supported?
        for i in range(len(X_test)):
            if isinstance(y_pred[i], str) and "," in y_pred[i]:
                y_pred[i] = y_pred[i].split(",")
            if isinstance(y_pred[i], list):
                final_str = X_test[i]
                for elem in y_pred[i]:
                    final_str = final_str.replace(tokenizer_mask, elem.strip(), 1)
                y_pred[i] = final_str
            else:
                y_pred[i] = X_test[i].replace(tokenizer_mask, y_pred[i].strip())
        y_pred = pd.DataFrame(y_pred)
        if indexes is not None:
            y_pred.index = indexes
        return y_pred

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        # if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
        #     X_test = X_test[X_test.columns[0]].to_numpy()
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError as e:
            logger.warning(f"TypeError exception raised. Reason: {e}")
            y_pred = self.model.predict(X_test)
        except RuntimeError as e:
            logger.warning(f"RuntimeError exception raised. Reason: {e}")
            return self.handle_device_failure(X_test, **kwargs)
        y_pred = self.return_masked_text(X_test, y_pred)
        return y_pred
