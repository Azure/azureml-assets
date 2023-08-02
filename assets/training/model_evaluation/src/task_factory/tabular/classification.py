# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Classification predictor."""

import numpy as np

from task_factory.base import PredictProbaWrapper, PredictWrapper
from functools import partial
from logging_utilities import get_logger

logger = get_logger(name=__name__)


class TabularClassifier(PredictWrapper, PredictProbaWrapper):
    """Tabular Classifier.

    Args:
        PredictWrapper (_type_): _description_
        PredictProbaWrapper (_type_): _description_
    """

    def _alternate_predict(self, X_test, **kwargs):
        """Model predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        device = kwargs.get("device", -1)
        multilabel = kwargs.get("multilabel", False)
        if self.is_torch or self.is_hf:
            preds = self.model.predict(X_test, device)
        else:
            preds = self.model.predict(X_test)
        if len(preds.shape) > 1:
            if multilabel:
                preds = np.where(preds > 0.5, 1, 0)
            preds = np.argmax(preds, axis=1)
        else:
            preds = (preds >= 0.5).all(1)
        return preds

    def _extract_predict_fn(self):
        """Extract_predict.

        Returns:
            _type_: _description_
        """
        if self.is_torch:
            return self._alternate_predict, self.model.predict

        predict_fn = self.model.predict
        predict_proba_fn = None

        raw_model = self.model._model_impl
        if raw_model is not None:
            predict_fn = raw_model.predict
            predict_proba_fn = getattr(raw_model, "predict_proba", None)

            try:
                import xgboost

                if isinstance(raw_model, xgboost.XGBModel):
                    predict_fn = partial(predict_fn, validate_features=False)
                    if predict_proba_fn is not None:
                        predict_proba_fn = partial(predict_proba_fn, validate_features=False)
            except Exception:
                pass

        if predict_proba_fn is None:
            predict_fn = self._alternate_predict
            predict_proba_fn = raw_model.predict

        return predict_fn, predict_proba_fn

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_transformer = kwargs.get("y_transformer", None)

        predict_fn, _ = self._extract_predict_fn()
        try:
            y_pred = predict_fn(X_test, **kwargs)
        except TypeError:
            y_pred = predict_fn(X_test)
        except RuntimeError:
            y_pred = self.handle_device_failure(X_test, **kwargs)
        if y_transformer is not None:
            y_pred = y_transformer.transform(y_pred).toarray()

        return y_pred

    def predict_proba(self, X_test, **kwargs):
        """Get prediction probabilities.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, pred_proba_fn = self._extract_predict_fn()
        if self.is_torch or self.is_hf:
            try:
                y_pred_proba = pred_proba_fn(X_test, **kwargs)
            except RuntimeError:
                y_pred_proba = self.handle_device_failure(X_test, **kwargs)
        else:
            y_pred_proba = pred_proba_fn(X_test)
        return y_pred_proba
