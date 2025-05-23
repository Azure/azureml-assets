# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text NER."""

from task_factory.base import PredictWrapper
import pandas as pd
import ast
from logging_utilities import get_logger
from constants import MODEL_FLAVOR

logger = get_logger(name=__name__)


def ner_predictor_wrapper_for_hftransformers(transformers_class):
    """NER predictor for hftransformers."""

    def ner_predictor_for_transformers(X_test, params=None):
        """NER predictor for transformers.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            transformers_class._override_model_config(params)
        except AttributeError:
            logger.info("Using newer version of mlflow.transformers._TransformersWrapper\
                        model config override API")
            transformers_class._merge_model_config_with_params(transformers_class.model_config, params)
        from azureml.evaluate.mlflow.hftransformers._task_based_predictors import NERPredictor
        predictor = NERPredictor(task_type="token-classification", model=transformers_class.pipeline.model,
                                 tokenizer=transformers_class.pipeline.tokenizer,
                                 config=transformers_class.pipeline.model.config)
        return predictor.predict(X_test, **transformers_class.model_config)

    return ner_predictor_for_transformers


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
        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            self.model._predict_fn = ner_predictor_wrapper_for_hftransformers(self.model._model_impl)
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError as e:
            logger.warning(f"TypeError exception raised. Reason: {e}")
            y_pred = self.model.predict(X_test)
        except RuntimeError as e:
            logger.warning(f"RuntimeError exception raised. Reason: {e}")
            return self.handle_device_failure(X_test, **kwargs)

        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred[y_pred.columns[0]].values
        if self.model_flavor == MODEL_FLAVOR.HFTRANSFORMERSV2 or self.model_flavor == MODEL_FLAVOR.HFTRANSFORMERS \
                or self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            y_pred = list(map(lambda x: ast.literal_eval(x), y_pred))
        return y_pred
