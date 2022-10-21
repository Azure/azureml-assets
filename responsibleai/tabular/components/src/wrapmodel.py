# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import argparse
import mlflow
import pandas as pd
import numpy as np

from pathlib import Path
from typing import List




def _get_logger(name):
    logging.basicConfig(filename="./outputs/wrapmodel.log",
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    _logger = logging.getLogger(name)

    return _logger
# logging.basicConfig(level=logging.INFO)
_logger = _get_logger(__file__)

def ensure_list(input) -> List:
    if isinstance(input, list):
        _logger.info("input was list")
        return input
    else:
        _logger.info(f"Converting {type(input)} to list")
        return list(input)


# Define the model class
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, target_mlflow_dir: str, wd_dir: str):
        self._target_mlflow_path = target_mlflow_dir
        self._model = None  # Lazy load
        self.logger = self._get_logger("wrap_model_mlflow_loaded")
        self.wd_dir = wd_dir
        self.logger.info("Created ModelWrapper for path: {0}".format(target_mlflow_dir))

    def _load_model(self):
        if self._model is None:
            self.logger.info(
                f"Loading wrapped model with mlflow: {self._target_mlflow_path}"
            )
            self._model = mlflow.pyfunc.load_model(self._target_mlflow_path)._model_impl

    def predict(self, context, X):
        self._load_model()
        self.logger.info("Calling predict and predict_proba")
        self.logger.info(f"Input X is type {type(X)}")
        preds = self._call_model("predict", X)
        pred_probas = self._call_model("predict_proba", X)

        self.logger.info(f"Preds is of type {type(preds)} {preds} ")
        self.logger.info(f"Pred_probas is of type {type(pred_probas)} {pred_probas}")
        result = {
            "pred": ensure_list(preds),
            "pred_proba": ensure_list(pred_probas),
        }
        return result

    def _call_model(self, method_name: str, X):
        self._load_model()
        if hasattr(self._model, method_name):
            method = getattr(self._model, method_name)
            return method(X)
        else:
            return []

    @staticmethod
    def wrap_mlflow_model(target_mlflow_dir: str, wrapped_model_dir: str):
        _logger.info("target_mlflow_dir: {0}".format(target_mlflow_dir))
        _logger.info("Target directory: {0}".format(wrapped_model_dir))

        mlflow_dirname = Path(target_mlflow_dir).resolve()
        conda_file = str(mlflow_dirname / "conda.yaml")

        wrapped_model = ModelWrapper(str(target_mlflow_dir), os.getcwd())

        _logger.info("Invoking mlflow.pyfunc.save_model")
        mlflow.pyfunc.save_model(
            path=wrapped_model_dir, python_model=wrapped_model, conda_env=conda_file
        )

    def _get_logger(self, name):
        logging.basicConfig(filename=os.path.join(self.wd_dir,"/outputs/wrapmodel_mlflow_loaded.log"),
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        _logger = logging.getLogger(name)
        return _logger


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--wrapped_model_path", type=str, required=True)
    # parse args
    args = parser.parse_args()

    # return args
    return args


args = parse_args()
ModelWrapper.wrap_mlflow_model(
    target_mlflow_dir=args.model_path, wrapped_model_dir=args.wrapped_model_path
)
