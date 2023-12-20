# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base Predictor."""

from abc import abstractmethod, ABC

import torch
from logging_utilities import get_logger
import azureml.evaluate.mlflow as aml_mlflow
import numpy as np
import pandas as pd
import os
import constants

from utils_load import load_model

logger = get_logger(name=__name__)


class BasePredictor(ABC):
    """Abstract Class for Predictors."""

    def __init__(self, model_uri, task_type, device=None):
        """__init__.

        Args:
            mlflow_model (_type_): _description_
        """
        self.model, self.model_flavor = load_model(model_uri=model_uri, device=device, task=task_type)

        is_torch, is_hf = False, False
        if self.model.metadata.flavors.get(aml_mlflow.pytorch.FLAVOR_NAME):
            is_torch = True
        if self.model_flavor in constants.ALL_MODEL_FLAVORS:
            is_hf = True

        if is_torch:
            self.model = self.model._model_impl
        self.model_uri = model_uri
        self.task_type = task_type
        self.current_device = device
        self.device = device
        self.is_torch = is_torch
        self.is_hf = is_hf
        super().__init__()

    def _ensure_base_model_input_schema(self, X_test):
        # todo: get input_schema for transformers
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
                        key = list(X_test.keys())[0]
                        X_test[input_names[0]] = X_test[key]
                        X_test.pop(key)

    def _ensure_model_on_cpu(self):
        """Ensure model is on cpu.

        Args:
            model (_type_): _description_
        """
        if self.is_hf:
            if hasattr(self.model._model_impl, "hf_model"):
                self.model._model_impl.hf_model = self.model._model_impl.hf_model.cpu()
            else:
                logger.warning("hf_model not found in mlflow model")
        elif self.is_torch:
            import torch
            if isinstance(self.model, torch.nn.Module):
                self.model = self.model.cpu()
            elif hasattr(self.model, "_model_impl") and isinstance(self.model._model_impl, torch.nn.Module):
                self.model._model_impl = self.model._model_impl.cpu()
            else:
                logger.warning("Torch model is not of type nn.Module")

    def _get_model_device(self):
        """Fetch Model device."""
        device = None
        if self.is_hf:
            if hasattr(self.model._model_impl, "hf_model"):
                device = self.model._model_impl.hf_model.device
            else:
                logger.warning("hf_model not found in mlflow model")
        elif self.is_torch:
            import torch
            if isinstance(self.model, torch.nn.Module):
                device = self.model.device
            elif hasattr(self.model, "_model_impl") and isinstance(self.model._model_impl, torch.nn.Module):
                device = self.model._model_impl.device
            else:
                logger.warning("Torch model is not of type nn.Module")
        return device

    def handle_device_failure(self, X_test, **kwargs):
        """Handle device failure."""
        if self.task_type == constants.TASK.SUMMARIZATION or \
                self.task_type == constants.TRANSFORMERS_TASK.SUMMARIZATION:
            logger.info("Reloading the model in a single device for summarization task.")
            os.environ["MLFLOW_HUGGINGFACE_USE_DEVICE_MAP"] = "False"
        predict_fn = kwargs.get('predict_fn', self.model.predict)
        if self.device == constants.DEVICE.AUTO and torch.cuda.is_available():
            try:
                cuda_current_device = torch.cuda.current_device()
                logger.info("Loading model and prediction with cuda current device.")
                if self.current_device != cuda_current_device:
                    logger.info(
                        f"Current Device: {self.current_device} does not match expected device {cuda_current_device}")
                    self.model, _ = load_model(self.model_uri, cuda_current_device, self.task_type)
                    self.current_device = cuda_current_device
                    predict_fn = kwargs.get('predict_fn', self.model.predict)
                kwargs["device"] = self.current_device
                return predict_fn(X_test, **kwargs)
            except TypeError as e:
                logger.warning("Failed on GPU with error: " + repr(e))
                logger.info("Trying on GPU without kwargs.")
                return predict_fn(X_test)
            except Exception as e:
                logger.warning("Failed on GPU with error: " + repr(e))
        if self.device != -1:
            logger.warning("Running predictions on CPU.")
            self.model, _ = load_model(self.model_uri, -1, self.task_type)
            predict_fn = kwargs.get('predict_fn', self.model.predict)
            try:
                logger.info("Loading model and prediction with cuda current device. Trying CPU.")
                if 'device' in kwargs:
                    kwargs.pop('device')
                if 'device_map' in kwargs:
                    kwargs.pop('device_map')
                return predict_fn(X_test, **kwargs)
            except TypeError as e:
                logger.warning("Failed on CPU with error: " + repr(e))
                logger.info("Trying on CPU without kwargs.")
                return predict_fn(X_test)
            except Exception as e:
                logger.warning("Failed on CPU with error: " + repr(e))
                raise e


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

    @abstractmethod
    def rolling_forecast(self, X_test, step=1):
        """Abstract forecast.

        Args:
            X_test (_type_): _description_
            step (_type_): _description_
        """
        pass
