# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import abstractmethod
import ast
import numpy as np
import pandas as pd

from azureml.metrics import compute_metrics, constants
from azureml.metrics.constants import Metric

from constants import TASK


class EvaluatorFactory:
    """Evaluator Factory Class."""

    def __init__(self):
        """
        Initialize evaluator factory and register evaluators for each task type.
        """
        self._evaluators = {
            TASK.CLASSIFICATION: ClassifierEvaluator,
            TASK.CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            TASK.REGRESSION: RegressorEvaluator,
        }

    def get_evaluator(self, task_type, metrics_config=None):
        """
        Get evaluator based on task type.
        Args:
            task_type: task type
            metrics_config: metrics config

        Returns: evaluator object

        """
        if metrics_config is None:
            metrics_config = {}
        return self._evaluators[task_type](task_type, metrics_config)


class Evaluator:
    """Evaluator for Compute Metrics mode."""

    def __init__(self, task_type, metrics_config):
        """
        Initialize evaluator.
        Args:
            task_type: string
            metrics_config: dict
        """
        self.task_type = task_type
        self.metrics_config = metrics_config

    @abstractmethod
    def evaluate(self, metrics_dto, **kwargs):
        """Evaluate predictions.

        Args:
            metrics_dto: metrics dto

        Returns:
            metrics: dict
        """
        pass

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds: predictions

        Returns: np array of predictions

        """
        if isinstance(preds, pd.DataFrame) and len(preds.columns) == 1:
            return preds[preds.columns[0]].to_numpy()
        if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
            return preds.to_numpy()
        if isinstance(preds, list):
            return np.array(preds)
        return preds


class ClassifierEvaluator(Evaluator):

    def __init__(self, task_type, metrics_config):
        """
        Initialize classifier evaluator.
        Args:
            task_type: string
            metrics_config: dict
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, metrics_dto, **kwargs):
        """Evaluate classification.

        Args:
            metrics_dto: metrics dto

        Returns: metrics
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
            task_type: string
            metrics_config: dict
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, metrics_dto, **kwargs):
        """

        Args:
            metrics_dto: metrics dto
            **kwargs:

        Returns: metrics

        """
        y_pred = self._convert_predictions(metrics_dto.predictions)
        y_test = self._convert_predictions(metrics_dto.ground_truth)
        if "metrics" not in self.metrics_config:
            self.metrics_config["metrics"] = [Metric.RMSE, Metric.MeanAbsError]
        metrics = compute_metrics(task_type=constants.Tasks.REGRESSION, y_test=y_test, y_pred=y_pred
                                  , **self.metrics_config)
        return metrics


class ClassifierMultilabelEvaluator(Evaluator):
    """Classifier Multilabel.
    """

    def __init__(self, task_type, metrics_config):
        """
        Args:
            task_type: string
            metrics_config: dict
        """
        super().__init__(task_type, metrics_config)

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds: predictions

        Returns: np array of predictions
        """
        preds = super()._convert_predictions(preds)
        if hasattr(preds, "ndim") and preds.ndim == 1 and len(preds) > 0 and isinstance(preds[0], str):
            preds = np.array(list(map(lambda x: ast.literal_eval(x), preds)))
        return preds

    def evaluate(self, metrics_dto, **kwargs):
        """
        Evaluate multilabel classification.
        Args:
            metrics_dto: Metrics DTO

        Returns:
            metrics: dict
        """
        y_pred = self._convert_predictions(metrics_dto.predictions)
        y_test = self._convert_predictions(metrics_dto.ground_truth)
        if "metrics" not in self.metrics_config:
            self.metrics_config["metrics"] = [Metric.Accuracy, Metric.PrecisionMacro, Metric.RecallMacro,
                                              Metric.F1Macro]
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION_MULTILABEL, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics
