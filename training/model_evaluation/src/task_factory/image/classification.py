# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image classification predictor."""

import ast
import numpy as np
import pandas as pd

from typing import Dict, List, Union

from task_factory.tabular.classification import TabularClassifier


def _convert_predictions(preds: Union[pd.DataFrame, pd.Series, list, np.ndarray]) -> np.ndarray:
    """Convert predictions to numpy array.

    Args:
        predictions(pd.DataFrame, pd.Series, list, np.ndarray): predictions

    Returns:
        np.ndarray: numpy array of predictions
    """
    if isinstance(preds, pd.DataFrame) and len(preds.columns) == 1:
        return preds[preds.columns[0]].to_numpy()
    if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
        return preds.to_numpy()
    if isinstance(preds, list):
        return np.array(preds)
    return preds


class ImageMulticlassClassifier(TabularClassifier):
    """Image multiclass classifier."""

    # Todo: Check if ImageClassificationPredictor is required in azureml-evaluate-mlflow in _task_based_predictors.py

    def predict(self, x_test, **kwargs) -> np.ndarray:
        """Get predicted labels.

        Args:
            x_test (pd.DataFrame): Input images for prediction. Pandas DataFrame with a first column name
                ["image"] of images where each image is in base64 String format.
        Returns:
            np.ndarray: numpy array of predicted labels.
        """
        # Image classification predict() returns both labels and probs
        op_df = super().predict(x_test, **kwargs)
        y_pred = _convert_predictions(op_df["labels"])
        return y_pred

    def predict_proba(self, x_test, **kwargs) -> List[Dict[str, float]]:
        """Get predicted probabilities.

        Args:
            x_test (pd.DataFrame): Input images for prediction. Pandas DataFrame with a first column name
                ["image"] of images where each image is in base64 String format.
        Returns:
            List[Dict[str, float]]: list containing dictionary of predicted probabilities like below example.
                [{"0": 0.1, "1": 0.2, "2": 0.7}, {}, {}, ...]
        """
        # Image classification predict() returns both labels and probs
        op_df = super().predict(x_test, **kwargs)
        y_pred_proba = _convert_predictions(op_df["probs"])
        # return a list of dictionary
        return [
            {str(i): prob_instance for i, prob_instance in enumerate(prob_instance)}
            for prob_instance in y_pred_proba
        ]


class ImageMultilabelClassifier(TabularClassifier):
    """Image multilabel classifier."""

    # Todo: Check if ImageClassificationPredictor is required in azureml-evaluate-mlflow in _task_based_predictors.py

    def predict(self, x_test, **kwargs) -> List[List[str]]:
        """Get predicted labels.

        Args:
            x_test (pd.DataFrame): Input images for prediction. Pandas DataFrame with a first column name
                ["image"] of images where each image is in base64 String format.
        Returns:
            List[str]: List of list of predicted labels.
        """
        # Image classification predict() returns both labels and probs
        y_pred = super().predict(x_test, **kwargs)["labels"]
        y_pred = [str(x) for x in y_pred]
        # Converting to list of list of strings
        y_pred = list(map(lambda x: ast.literal_eval(x), y_pred))
        return y_pred

    def predict_proba(self, x_test, **kwargs) -> List[Dict[str, float]]:
        """Get predicted probabilities.

        Args:
            x_test (pd.DataFrame): Input images for prediction. Pandas DataFrame with a first column name
                ["image"] of images where each image is in base64 String format.
        Returns:
            List[Dict[str, float]]: list containing dictionary of predicted probabilities like below example.
                [{"0": 0.1, "1": 0.2, "2": 0.7}, {}, {}, ...]
        """
        # Image classification predict() returns both labels and probs
        op_df = super().predict(x_test, **kwargs)
        y_pred_proba = _convert_predictions(op_df["probs"])
        # return a list of dictionary
        return [
            {str(i): prob_instance for i, prob_instance in enumerate(prob_instance)}
            for prob_instance in y_pred_proba
        ]
