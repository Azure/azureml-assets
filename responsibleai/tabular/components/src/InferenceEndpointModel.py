# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import shutil
import requests
import pandas as pd
import numpy as np
import mltable

from cmd_util import EndpointUtil

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

def PrintDirTree(rootDir): 
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        print(path)
        if os.path.isdir(path): 
            PrintDirTree(path)

class InferenceEndpointModel:
    def __init__(self, model_type, model_path, wrapped_model_path):
        self._model_path = model_path
        self._server_initialized = False
        self._server_endpoint = None

        self.cmd_helper = EndpointUtil(
            model_path=model_path, wrapped_model_path=wrapped_model_path
        )

        if model_type != "classification":
            del self.__class__.predict_proba

    def _initialize_server(self):
        _logger.info("Initlizing local inference endpoint.")
        _logger.info("Creating conda environment.")
        success, output = self.cmd_helper.create_conda_environment()
        if not success:
            _logger.info(
                f"Non zero exit code creating conda environment, error is: {output}"
            )

        success, output = self.cmd_helper.create_wrapped_model()
        if not success:
            _logger.info(
                f"Non zero exit code creating wrapped model, error is: {output}"
            )

        _logger.info(f'Working directory:{os.getcwd()}')
        _logger.info(f'Dir tree:{PrintDirTree(os.getcwd())}')

        success, endpoint = self.cmd_helper.create_endpoint()
        self._server_endpoint = endpoint
        self._server_initialized = True
        _logger.info(f"Initilized endpoint at {endpoint}")

    def _convert_to_json(self, input) -> str:
        result = ""
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.Series):
            result = input.to_json(orient="split")
        else:
            raise RuntimeError(
                "Inference endpoint model only support input of pandas dataframe or pandas series."
            )

        return result

    def _call_model_and_extract(self, input_data, target: str):
        if not self._server_initialized:
            self._initialize_server()

        payload = self._convert_to_json(input_data)

        headers = {"Content-Type": "application/json"}
        r = requests.post(
            f"{self._server_endpoint}/invocations",
            headers=headers,
            data=payload,
            timeout=100,
        )

        if r.status_code != 200:
            raise RuntimeError(
                "Inference endpoint did not return successful result, status code: {}, message: {}".format(
                    r.status_code, r.text
                )
            )

        decoded = json.loads(r.text)
        result = np.asarray(decoded[target])

        return result

    def predict(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred")

    def predict_proba(self, input_df: pd.DataFrame):
        return self._call_model_and_extract(input_df, "pred_proba")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d