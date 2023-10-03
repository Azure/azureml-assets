# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for convert the results to dict."""

import copy
from typing import Any, Dict, Optional
import pandas as pd


class ResultConverters:
    """Convert the batch inference output to different results."""

    TOKEN_KEYS = ["completion_tokens", "prompt_tokens", "total_tokens"]
    LATENCY_KEYS = ["start", "end", "latency"]
    METADATA_KEY_IN_RESULT = 'request_metadata'
    PREDICTION_COL_NAME = 'prediction'

    def __init__(
            self, model_type: str, metadata_key: str, data_id_key: str,
            label_key: str, ground_truth_df: Optional[pd.DataFrame], fallback_value: str
    ) -> None:
        """Init for the result converter."""
        self._model_type = model_type
        self._metadata_key = metadata_key
        self._label_key = label_key
        self._data_id_key = data_id_key
        self._lookup_dict = {}
        self._fallback_value = fallback_value
        if ground_truth_df is not None and self._is_aoai_model():
            print("receive ground truth columns {}".format(ground_truth_df.columns))
            for index, row in ground_truth_df.iterrows():
                self._lookup_dict[row[self._data_id_key]] = row[self._label_key]

    def convert_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the input result to predictions."""
        if not self.is_result_success(result):
            return self._get_fallback_output()
        return self._get_raw_output(result)

    def convert_result_perf(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the result to perf metrics."""
        usage = copy.deepcopy(result.get('usage', {}))
        for key in ResultConverters.TOKEN_KEYS:
            if key not in usage:
                usage[key] = -1
        for key in ResultConverters.LATENCY_KEYS:
            usage[key] = result.get(key, -1)
        return usage

    def convert_result_ground_truth(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the result to ground truth."""
        ground_truth = ''
        if self._is_llama_model():
            if self._metadata_key:
                ground_truth = self._get_request(result)[self._metadata_key][self._label_key]
            else:
                ground_truth = result[self.METADATA_KEY_IN_RESULT][self._label_key]
        elif self._is_aoai_model():
            for k, v in self._lookup_dict.items():
                if k in self._get_request(result):
                    ground_truth = v
        return {'ground_truth': ground_truth}

    def _get_raw_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        prediction = ''
        if self._is_llama_model():
            prediction = ResultConverters._get_oss_response_result(result)
        elif self._is_aoai_model():
            prediction = ResultConverters._get_aoai_response_result(result)
        return {ResultConverters.PREDICTION_COL_NAME: prediction}

    def _is_llama_model(self) -> bool:
        return self._model_type.lower() == "llama"

    def _is_aoai_model(self) -> bool:
        return self._model_type.lower() == "gpt"
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        return {ResultConverters.PREDICTION_COL_NAME: self._fallback_value}
    
    def is_result_success(self, result: Dict[str, Any]) -> bool:
        """Check if the result contains a successful response."""
        if result['status'].lower() != "success":
            return False
        # handle the scenario the 200 with failure in response.
        try:
            _ = self._get_raw_output(result)
        except Exception as e:
            print(f'Converting meet errors {e}')
            return False
        return True

    @staticmethod
    def _get_request(result: Dict[str, Any]) -> Any:
        return result['request']

    @staticmethod
    def _get_raw_prompt(result: Dict[str, Any]) -> Any:
        return ResultConverters._get_request(result)['input_data']['input_string']

    @staticmethod
    def _get_response(result: Dict[str, Any]) -> Any:
        return result['response']

    @staticmethod
    def _get_aoai_response_result(result: Dict[str, Any]) -> Any:
        response = ResultConverters._get_response(result)
        return response["choices"][0]["message"]["content"]

    @staticmethod
    def _get_oss_response_result(result: Dict[str, Any]) -> Any:
        response = ResultConverters._get_response(result)
        print(f"response is {response}")
        if isinstance(response, str):
            return response
        if isinstance(response, list):
            if "0" in response[0]:
                return response[0]["0"]
            return response[0]['output']
        if '0' in response:
            return response['0']
        return response['output']
