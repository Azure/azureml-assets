# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluator."""

import ast
from abc import abstractmethod

from constants import TASK, ForecastingConfigContract
import pandas as pd
import numpy as np
from azureml.metrics import compute_metrics, constants
from logging_utilities import get_logger
from azureml.evaluate.mlflow.constants import ForecastColumns

logger = get_logger(name=__name__)


class EvaluatorFactory:
    """Evaluator Factory Class."""

    def __init__(self):
        """__init__."""
        self._evaluators = {
            TASK.CLASSIFICATION: ClassifierEvaluator,
            TASK.CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            "multiclass": ClassifierEvaluator,
            "tabular-regressor": RegressorEvaluator,
            TASK.REGRESSION: RegressorEvaluator,
            TASK.NER: NerEvaluator,
            "text-ner": NerEvaluator,
            TASK.TEXT_CLASSIFICATION: TextClassifierEvaluator,
            TASK.TEXT_CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            TASK.TRANSLATION: TranslationEvaluator,
            TASK.SUMMARIZATION: SummarizationEvaluator,
            TASK.QnA: QnAEvaluator,
            TASK.FILL_MASK: FillMaskEvaluator,
            TASK.TEXT_GENERATION: TextGenerationEvaluator,
            TASK.FORECASTING: ForecastingEvaluator,
        }

    def get_evaluator(self, task_type, metrics_config=None):
        """Get evaluator.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return self._evaluators[task_type](task_type, metrics_config)

    def register(self, name, obj):
        """Register evaluator.

        Args:
            name (_type_): _description_
            obj (_type_): _description_
        """
        self._evaluators[name] = obj


class Evaluator:
    """Evaluator for Compute Metrics mode."""

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        self.task_type = task_type
        self.metrics_config = metrics_config

    @abstractmethod
    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate predictions.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        pass

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(preds, pd.DataFrame) and len(preds.columns) == 1:
            return preds[preds.columns[0]].to_numpy()
        if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
            return preds.to_numpy()
        if isinstance(preds, list):
            return np.array(preds)
        return preds


class ClassifierEvaluator(Evaluator):
    """Classifier Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, y_pred_proba=None):
        """Evaluate classification.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.CLASSIFICATION, y_test=y_test, y_pred=y_pred,
                                  y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class TextClassifierEvaluator(Evaluator):
    """Text Classifier Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, y_pred_proba=None):
        """Evaluate classification.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION, y_test=y_test, y_pred=y_pred,
                                  y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class RegressorEvaluator(Evaluator):
    """Regressor Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate regression.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.REGRESSION, y_test=y_test, y_pred=y_pred,
                                  **self.metrics_config)
        return metrics


class ClassifierMultilabelEvaluator(Evaluator):
    """Classifier Multilabel.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        preds = super()._convert_predictions(preds)
        if hasattr(preds, "ndim") and preds.ndim == 1 and len(preds) > 0 and isinstance(preds[0], str):
            preds = np.array(list(map(lambda x: ast.literal_eval(x), preds)))
        return preds

    def evaluate(self, y_test, y_pred, y_pred_proba=None):
        """Evaluate multilabel.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION_MULTILABEL, y_test=y_test,
                                  y_pred=y_pred, y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class NerEvaluator(Evaluator):
    """NER Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def _convert_predictions(self, preds):
        """Convert predictions.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        preds = super()._convert_predictions(preds)
        if hasattr(preds, "ndim") and preds.ndim == 1 and len(preds) > 0 and isinstance(preds[0], str):
            preds = np.array(list(map(lambda x: ast.literal_eval(x), preds)))
        return preds

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate NER.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_NER, y_test=y_test, y_pred=y_pred,
                                  **self.metrics_config)
        return metrics


class SummarizationEvaluator(Evaluator):
    """Summarization Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate summarizer.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.SUMMARIZATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class QnAEvaluator(Evaluator):
    """QnA Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate QnA.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()

        # y_pred = []
        # for pred in y_pred:
        #     if (isinstance(pred, list) or isinstance(pred, np.ndarray)):
        #         if len(pred) >= 1:
        #             y_pred.append(pred[0])
        #     else:
        #         y_pred.append(pred)
        # logger.warning("Multiple ground truths are not supported for question-answering task currently.\
        #                 Considering only the first ground truth in case of multiple values.")
        metrics = compute_metrics(task_type=constants.Tasks.QUESTION_ANSWERING, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics


class TranslationEvaluator(Evaluator):
    """Translation Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Translation.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.TRANSLATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class FillMaskEvaluator(Evaluator):
    """Fill Mask Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Fill Mask.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()
        metrics = compute_metrics(task_type=constants.Tasks.FILL_MASK, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics


class TextGenerationEvaluator(Evaluator):
    """Text Generation Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate TextGeneration.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_GENERATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class ForecastingEvaluator(Evaluator):
    """Forecasting Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, X_test, **kwargs):
        """Evaluate regression.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)

        # In the evaluation component we do not have the model, so we cannot aggregate data.
        # In case user provided X_train or y train we will remove it from kwargs.
        kwargs.pop('X_train', None)
        kwargs.pop('y_train', None)
        # Take forecasting-specific parameters from the model.
        kwargs["time_series_id_column_names"] = self.metrics_config.get(
            ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES)
        kwargs["time_column_name"] = self.metrics_config.get(
            ForecastingConfigContract.TIME_COLUMN_NAME)
        metrics = compute_metrics(
            task_type=constants.Tasks.FORECASTING,
            y_test=y_test,
            y_pred=y_pred,
            X_test=X_test,
            **kwargs
        )
        X_test[ForecastColumns._ACTUAL_COLUMN_NAME] = y_test
        X_test[ForecastColumns._FORECAST_COLUMN_NAME] = y_pred
        return metrics
