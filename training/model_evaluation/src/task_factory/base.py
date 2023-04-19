# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base Predictor."""

from abc import abstractmethod, ABC
import azureml.evaluate.mlflow as aml_mlflow
import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Abstract Class for Predictors."""

    def __init__(self, mlflow_model):
        """__init__.

        Args:
            mlflow_model (_type_): _description_
        """
        is_torch, is_hf = False, False
        if mlflow_model.metadata.flavors.get(aml_mlflow.pytorch.FLAVOR_NAME):
            is_torch = True
        if mlflow_model.metadata.flavors.get(aml_mlflow.hftransformers.FLAVOR_NAME):
            is_hf = True

        if is_torch:
            self.model = mlflow_model._model_impl
        else:
            self.model = mlflow_model
        self.is_torch = is_torch
        self.is_hf = is_hf
        super().__init__()

    def _ensure_base_model_input_schema(self, X_test):
        input_schema = self.model.metadata.get_input_schema()
        if self.is_hf and input_schema is not None:
            if input_schema.has_input_names():
                # make sure there are no missing columns
                input_names = input_schema.input_names()
                expected_cols = set(input_names)
                # Hard coding logic for converting data to input string for base models
                if len(expected_cols) == 1 and input_names[0] == "input_string":
                    if isinstance(X_test, np.ndarray):
                        X_test = {input_names[0]: X_test}
                    elif isinstance(X_test, pd.DataFrame) and len(X_test.columns) == 1:
                        X_test.columns = input_names
                    elif isinstance(X_test, dict) and len(X_test.keys()) == 1:
                        key = X_test.keys()[0]
                        X_test[input_names[0]] = X_test[key]
                        X_test.pop(key)


class PredictWrapper(BasePredictor):
    """Abstract class for predict based models."""

    @abstractmethod
    def predict(self, X_test, **kwargs):
        """Abstract predict.

        Args:
            X_test (_type_): _description_
        """
        pass


class PredictProbaWrapper(BasePredictor):
    """Abstract class for predict_proba based models."""

    @abstractmethod
    def predict_proba(self, X_test, **kwargs):
        """Abstract Predict proba.

        Args:
            X_test (_type_): _description_
        """
        pass


# For Future:
class ForecastWrapper(BasePredictor):
    """Abstract class for forecasting models."""

    @abstractmethod
    def forecast(self, X_test, y_context, **kwargs):
        """Abstract forecast.

        Args:
            X_test (_type_): _description_
            y_context (_type_): _description_
        """
        pass
