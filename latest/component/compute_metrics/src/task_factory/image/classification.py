# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Image classification predictor."""

import numpy as np
import pandas as pd

from typing import Dict, List, Union

from constants import MODEL_FLAVOR
from task_factory.tabular.classification import TabularClassifier
from exceptions import ScoringException
from logging_utilities import get_logger

logger = get_logger(name=__name__)


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


MC_OUTPUT_SIGNATURE_ERROR_MESSAGE = (
    "The output of the model predict function should return List of labels for each image (List[str]). "
    "The output of the model predict_proba function (optional) should return "
    "List of probabilities for each label (List[List[float]]). "
    "The outermost List is for each image in a batch."
)

ML_OUTPUT_SIGNATURE_ERROR_MESSAGE = (
    "The output of the model predict function should return List of only "
    "predicted labels for each image (List[List[str]]). "
    "The output of the model predict_proba function (optional) should return "
    "List of probabilities for all labels (List[List[float]]). "
    "The outermost List is for each image in a batch. The inner List is for each label in the label list."
)


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
        try:
            op_df = super().predict(x_test, **kwargs)
        except (TypeError, AttributeError, NameError) as ex:
            if self.is_hf:
                raise
            else:
                raise ScoringException(MC_OUTPUT_SIGNATURE_ERROR_MESSAGE) from ex

        if not self.is_hf:
            return op_df

        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            predicted_labels = [item["label"] for item in op_df[0]]
        else:
            probs, labels = _convert_predictions(op_df["probs"]), _convert_predictions(op_df["labels"])
            label_indexes = [np.argmax(np.array(prob)) for prob in probs]
            predicted_labels = [label_list[index] for index, label_list in zip(label_indexes, labels)]

        y_pred = _convert_predictions(predicted_labels)
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

        if not self.is_hf:
            try:
                return super().predict_proba(x_test, **kwargs)
            except (TypeError, AttributeError, NameError) as ex:
                logger.warning(
                    f"Error occured in predict_proba function: {str(ex)} {MC_OUTPUT_SIGNATURE_ERROR_MESSAGE}"
                )
            return None

        op_df = super().predict(x_test, **kwargs)

        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            y_pred = _convert_predictions(op_df)
            labels = sorted(set([item["label"] for pred in y_pred for item in pred]))
            probs = [[0]*len(labels) for _ in range(len(y_pred))]
            for line_idx, pred in enumerate(y_pred):
                for ele in pred:
                    probs[line_idx][labels.index(ele["label"])] = ele["score"]
        else:
            probs = _convert_predictions(op_df)
        # return a list of dictionary
        return [{str(i): prob_instance for i, prob_instance in enumerate(prob_instance)} for prob_instance in probs]


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
        try:
            y_pred = super().predict(x_test, **kwargs)
        except (TypeError, AttributeError, NameError) as ex:
            if self.is_hf:
                raise
            else:
                raise ScoringException(ML_OUTPUT_SIGNATURE_ERROR_MESSAGE) from ex
        if not self.is_hf:
            return y_pred
        threshold = kwargs.get("threshold", 0.5)
        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            y_pred = _convert_predictions(y_pred)
            predicted_labels = [[item["label"] for item in pred if item["score"] > threshold] for pred in y_pred]
        else:
            pred_probs, pred_labels = _convert_predictions(y_pred["probs"]), _convert_predictions(y_pred["labels"])

            predicted_labels = []
            for probs, labels in zip(pred_probs, pred_labels):
                # Iterate through each image's predicted probabilities.
                image_labels = []
                for index, prob in enumerate(probs):
                    if prob >= threshold:
                        image_labels.append(labels[index])
                predicted_labels.append(image_labels)

        return predicted_labels

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
        if not self.is_hf:
            try:
                return super().predict_proba(x_test, **kwargs)
            except (TypeError, AttributeError, NameError) as ex:
                logger.warning(
                    f"Error occured in predict_proba function: {str(ex)} {ML_OUTPUT_SIGNATURE_ERROR_MESSAGE}"
                )
            return None

        op_df = super().predict(x_test, **kwargs)
        if self.model_flavor == MODEL_FLAVOR.TRANSFORMERS:
            y_pred = _convert_predictions(op_df)
            labels = sorted(set([item["label"] for pred in y_pred for item in pred]))
            pred_probs = [[0]*len(labels) for _ in range(len(y_pred))]
            for line_idx, pred in enumerate(y_pred):
                for ele in pred:
                    pred_probs[line_idx][labels.index(ele["label"])] = ele["score"]
        else:
            pred_probs = _convert_predictions(op_df["probs"])

        # return a list of dictionary
        pred_probs_in_dict = [
            {str(i): prob_instance for i, prob_instance in enumerate(prob_instance)} for prob_instance in pred_probs
        ]
        return pred_probs_in_dict
