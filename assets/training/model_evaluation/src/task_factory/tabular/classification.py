# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Classification predictor."""

import numpy as np
import pandas as pd

from task_factory.base import PredictProbaWrapper, PredictWrapper
from functools import partial
from logging_utilities import get_logger
from constants import MODEL_FLAVOR


logger = get_logger(name=__name__)


class TabularClassifier(PredictWrapper, PredictProbaWrapper):
    """Tabular Classifier.

    Args:
        PredictWrapper (_type_): _description_
        PredictProbaWrapper (_type_): _description_
    """

    def _handle_multiple_columns(self, X_test, **kwargs):
        if not isinstance(X_test, pd.DataFrame):
            return X_test, kwargs
        if len(X_test.columns) == 1:
            return X_test[X_test.columns[0]].values.tolist(), kwargs
        if len(X_test.columns) == 2:
            # ToDo: Should this be added for data with 1 columns. Might be needed for some models
            kwargs["truncation"] = True
            kwargs["padding"] = True
            if "text_pair" not in X_test.columns or "text" not in X_test.columns:
                X_test.columns = ["text", "text_pair"]
            return X_test.to_dict(orient="records"), kwargs
        return X_test, kwargs

    def _alternate_predict(self, X_test, **kwargs):
        """Model predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Todo: Check why device is removed from .predict
        multilabel = kwargs.get("multilabel", False)
        device = kwargs.get("device", -1)
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
            # raw_model.predict = hfTRansformers wrapper predict which supports kwargs
            # Catch: Transformers model predict changes model.config and is static
            # self.model.predict -> pyfunc predict with no kwargs
            predict_proba_fn = raw_model.predict
            predict_fn = raw_model.predict \
                if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS else self._alternate_predict

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

        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            params = kwargs.get("params", {})
            params["return_all_scores"] = False
            # Todo: Handle for multiple columns
            X_test, params = self._handle_multiple_columns(X_test, **params)
            kwargs = {"params": params}
        try:
            y_pred = predict_fn(X_test, **kwargs)
        except TypeError as e:
            logger.warning(f"TypeError exception raised. Reason: {e}")
            y_pred = predict_fn(X_test)
        except RuntimeError as e:
            logger.warning(f"RuntimeError exception raised. Reason: {e}")
            y_pred = self.handle_device_failure(X_test, **kwargs)
        if y_transformer is not None:
            y_pred = y_transformer.transform(y_pred).toarray()

        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS and "label" in y_pred:
            return y_pred["label"]  # Current transformers return dataframse with labels and scores

        return y_pred

    def predict_proba(self, X_test, **kwargs):
        """Get prediction probabilities.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        _, pred_proba_fn = self._extract_predict_fn()

        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            params = kwargs.get("params", {})
            params["return_all_scores"] = True
            # Todo: Handle multiple columns
            X_test, kwargs = self._handle_multiple_columns(X_test, **kwargs)
            y_transformer = kwargs.pop("y_transformer", None)
            kwargs = {"params": params}
            try:
                y_pred = pred_proba_fn(X_test, **kwargs)
            except TypeError as e:
                logger.warning(f"TypeError exception raised. Reason: {e}")
                y_pred = pred_proba_fn(X_test)
            except RuntimeError as e:
                logger.warning(f"RuntimeError exception raised. Reason: {e}")
                y_pred = self.handle_device_failure(X_test, **kwargs)
            if y_transformer is not None:
                y_pred = y_transformer.transform(y_pred).toarray()
            y_pred_proba = y_pred.applymap(lambda x: x['score'] if 'score' in x else None)
            y_pred_proba.columns = y_pred.iloc[0].apply(lambda x: x['label'] if 'label' in x else None).to_list()

        else:
            if self.is_torch or self.is_hf:
                try:
                    y_pred_proba = pred_proba_fn(X_test, **kwargs)
                except RuntimeError:
                    y_pred_proba = self.handle_device_failure(X_test, **kwargs)
            else:
                y_pred_proba = pred_proba_fn(X_test)
        return y_pred_proba
