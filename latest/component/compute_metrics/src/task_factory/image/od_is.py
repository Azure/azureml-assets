# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Object detection and instance segmentation Predictor."""
from task_factory.base import PredictWrapper

from azureml.evaluate.mlflow.models.evaluation.azureml._image_od_is_evaluator import (
    ImageOdIsEvaluator,
)


class ImageOdIsPredictor(PredictWrapper):
    """Object detection/Instance Segmentation predictor."""

    def predict(self, X_test, **kwargs):
        """Object detection/Instance Segmentation inference.

        Args:
            x_test (pd.DataFrame): Input images for prediction. Pandas DataFrame with a first column name
                ["image"] of images where each image is in base64 String format.
        Returns:
            List[Dict[str, Any]]: list containing dictionary of predicted probabilities, boxes and masks etc
        """
        return ImageOdIsEvaluator.predict(model=self.model, X_test=X_test, **kwargs)
