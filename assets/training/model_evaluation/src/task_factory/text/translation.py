# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Text Translation."""

from task_factory.base import PredictWrapper
from logging_utilities import get_logger

logger = get_logger(name=__name__)


class Translator(PredictWrapper):
    """Translator.

    Args:
        PredictWrapper (_type_): _description_
    """

    def _validate_translation_langs(self, source_lang, target_lang):
        """Validate Translation languages.

        Args:
            source_lang (_type_): _description_
            target_lang (_type_): _description_
        """
        if len(source_lang) != 2:
            raise ValueError("Invalid language id passed for source language " + source_lang)

        if len(target_lang) != 2:
            raise ValueError("Invalid language id passed for target language " + target_lang)

    def predict(self, X_test, **kwargs):
        """Predict.

        Args:
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        self._ensure_base_model_input_schema(X_test)
        source_lang = kwargs.pop("source_lang", None)
        target_lang = kwargs.pop("target_lang", None)
        if self.is_hf and source_lang is not None and target_lang is not None:
            self._validate_translation_langs(source_lang, target_lang)
            task_type = "translation_" + source_lang + "_to_" + target_lang
            logger.info("Updating hf conf with task type" + task_type)
            kwargs["task_type"] = task_type
        try:
            y_pred = self.model.predict(X_test, **kwargs)
        except TypeError as e:
            logger.warning(f"TypeError exception raised. Reason: {e}")
            y_pred = self.model.predict(X_test)
        except RuntimeError as e:
            logger.warning(f"RuntimeError exception raised. Reason: {e}")
            return self.handle_device_failure(X_test, **kwargs)

        return y_pred
