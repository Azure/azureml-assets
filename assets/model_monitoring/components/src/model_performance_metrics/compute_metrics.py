# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluator."""

import ast
import pandas as pd
import numpy as np
from abc import abstractmethod

from constants import TASK

from azureml.metrics import compute_metrics, constants
from azureml.metrics.constants import Metric


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
    def evaluate(self, metrics_dto, **kwargs):
        """Evaluate predictions.

        Args:
            metrics_dto: _description_

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

    def evaluate(self, metrics_dto, **kwargs):
        """Evaluate classification.

        Args:
            metrics_dto (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(metrics_dto.predictions)
        y_test = self._convert_predictions(metrics_dto.ground_truth)
        if "metrics" not in self.metrics_config:
            self.metrics_config["metrics"] = [Metric.Accuracy, Metric.PrecisionMacro, Metric.RecallMacro,
                                              Metric.F1Macro]
        metrics = compute_metrics(task_type=constants.Tasks.CLASSIFICATION, y_test=y_test, y_pred=y_pred
                                  , **self.metrics_config)
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

    def evaluate(self, metrics_dto, **kwargs):
        """

        Args:
            metrics_dto: _description_
            **kwargs:

        Returns:

        """
        y_pred = self._convert_predictions(metrics_dto.predictions)
        y_test = self._convert_predictions(metrics_dto.ground_truth)
        if "metrics" not in self.metrics_config:
            self.metrics_config["metrics"] = [Metric.RMSE, Metric.MeanAbsError]  # Metric.MSE
        metrics = compute_metrics(task_type=constants.Tasks.REGRESSION, y_test=y_test, y_pred=y_pred
                                  , **self.metrics_config)
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

    def evaluate(self, metrics_dto, **kwargs):
        """Evaluate multilabel.

        Args:
            metrics_dto: _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(metrics_dto.predictions)
        y_test = self._convert_predictions(metrics_dto.ground_truth)
        if "metrics" not in self.metrics_config:
            self.metrics_config["metrics"] = [Metric.Accuracy, Metric.PrecisionMacro, Metric.RecallMacro,
                                              Metric.F1Macro]
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION_MULTILABEL, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics
