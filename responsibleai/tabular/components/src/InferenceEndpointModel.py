# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import requests
import pandas as pd
import numpy as np
import mlflow
import yaml

from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from cmd_util import EndpointUtil

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

COMMON_SERVING_ENDPOINT = "http://127.0.0.1:5000"

def PrintDirTree(rootDir): 
    for lists in os.listdir(rootDir): 
        path = os.path.join(rootDir, lists) 
        print(path)
        if os.path.isdir(path): 
            PrintDirTree(path)

def download_mlflow_model_artifacts(tracking_uri, model_uri):
    mlflow.set_tracking_uri(tracking_uri)
    return get_artifact_repository(artifact_uri=model_uri).download_artifacts(
        artifact_path="", dst_path="./")

class InferenceEndpointModel:
    def __init__(self, model_type, model_uri, mlflow_tracking_uri, model_path, wrapped_model_path):
        self._model_type = model_type
        self._model_uri = model_uri
        self._mlflow_tracking_uri = mlflow_tracking_uri
        self._server_endpoint = None
        self._wrapped_model_path = wrapped_model_path
        self._model_path = model_path
        self._model_signature = None
        self._had_successful_request = False

        self._set_model_signature()

        if model_type != "classification":
            del self.__class__.predict_proba

    def _initialize_server(self):
        try:
            requests.get(COMMON_SERVING_ENDPOINT)
            self._server_endpoint = COMMON_SERVING_ENDPOINT
            _logger.info("Inference endpoint already initlizied. Skip endpoint creation.")
            return
        except requests.ConnectionError:
            pass
        _logger.info("Initlizing local inference endpoint.")
        _logger.info("Creating conda environment.")

        if self._model_path is None:
            _logger.info("Model path is none, retrieving mlflow model with model uri.")
            self._model_path = download_mlflow_model_artifacts(self._mlflow_tracking_uri, self._model_uri)

        cmd_helper = EndpointUtil(
            model_path=self._model_path, wrapped_model_path=self._wrapped_model_path
        )

        success, output = cmd_helper.create_conda_environment()
        if not success:
            _logger.info(
                f"Non zero exit code creating conda environment, error is: {output}"
            )

        success, output = cmd_helper.create_wrapped_model()
        if not success:
            _logger.info(
                f"Non zero exit code creating wrapped model, error is: {output}"
            )

        _logger.info(f'Working directory:{os.getcwd()}')
        _logger.info(f'Dir tree:')
        PrintDirTree(os.getcwd())

        success, endpoint = cmd_helper.create_endpoint()
        self._server_endpoint = endpoint
        _logger.info(f"Initilized endpoint at {endpoint}")

    def _set_model_signature(self):
        with open(os.path.join(self._model_path, 'MLmodel'), 'r') as file:
            mlflow_yaml = yaml.safe_load(file)
        
        if "signature" in mlflow_yaml:
            self._model_signature = {}
            for s in json.loads(mlflow_yaml["signature"]["inputs"]):
                self._model_signature[s["name"]] = s["type"]

    def _convert_to_json(self, input) -> str:
        result = ""
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.Series):
            input = self._convert_to_schema(input)
            result = input.to_json(orient="split")
        elif isinstance(input, np.ndarray):
            result = json.dumps(input.tolist())
        else:
            _logger.warn("Received input of type {} which is not supported. This will likely result in error in calling mlflow model serving endpoint".format(type(input)))
            result = json.dumps(input)

        return result

    def _convert_to_schema(self, df) -> str:
        if self._model_signature is None:
            return df

        for cname, dtype in self._model_signature.items():
            if cname not in df.dtypes:
                continue
            if dtype == "string" and not pd.api.types.is_string_dtype(df[cname].dtype):
                df[cname] = df[cname].astype(str)
                continue
            if dtype == "long" and not pd.api.types.is_numeric_dtype(df[cname].dtype):
                df[cname] = df[cname].astype(int)
                continue
        
        return df

    def _call_model_and_extract(self, input_data, target: str):
        if self._server_endpoint is None:
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
                "Inference endpoint did not return successful result when called with {}, status code: {}, message: {}".format(
                    payload, r.status_code, r.text
                )
            )

        if not self._had_successful_request:
            _logger.info("First successful request to inference endpoint when called with {} with response {}".format(payload, r.text))

        self._had_successful_request = True

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
        self._server_endpoint = None
        self._model_path = None
        self._had_successful_request = False

        self._model_type = d["_model_type"]
        self._model_uri = d["_model_uri"]
        self._mlflow_tracking_uri = d["_mlflow_tracking_uri"]
        self._wrapped_model_path = d["_wrapped_model_path"]
        self._model_signature = d["_model_signature"]