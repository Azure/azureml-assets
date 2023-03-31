"""Base Predictor."""

from abc import abstractmethod, ABC
import azureml.evaluate.mlflow as aml_mlflow


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
