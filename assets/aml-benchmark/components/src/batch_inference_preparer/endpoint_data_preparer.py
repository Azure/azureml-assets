# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for endpoint data preparer."""

from typing import Any, Dict
import json
import re

from utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from utils.online_endpoint.endpoint_utils import EndpointUtilities


class EndpointDataPreparer:
    """Endpoint data preparer class."""

    PAYLOAD_HASH = "payload_id"
    PAYLOAD_GROUNDTRUTH = "label"

    def __init__(self, model_type: str, batch_input_pattern: str, label_key: str = None):
        """Init for endpoint data preparer."""
        self._model = OnlineEndpointModel(model_type=model_type, model=None, model_version=None)
        self._batch_input_pattern = batch_input_pattern
        self._label_key = label_key

    def convert_input_dict(self, origin_json_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert input dict to the corresponding payload."""
        return self._convert_python_pattern(origin_json_dict)

    def convert_ground_truth(
            self, origin_json_dict: Dict[str, Any], payload: Any
    ) -> Dict[str, Any]:
        """Convert the ground truth to the corresponding payload with id."""
        row_id = EndpointUtilities.hash_payload_prompt(payload, self._model)
        return {
            EndpointDataPreparer.PAYLOAD_HASH: row_id,
            EndpointDataPreparer.PAYLOAD_GROUNDTRUTH: origin_json_dict.get(self._label_key, ""),
        }

    def validate_output(self, output_payload_dict: Dict[str, Any]):
        """Validate the output payload."""
        errors = []
        if self._model.is_oss_model():
            if "input_data" not in output_payload_dict:
                errors.append("`input_data` should be presented in the payload json.")
            elif "input_string" not in output_payload_dict["input_data"]:
                errors.append(
                    "`input_string` should be presented in the `input_data` fields of payload json.")
            elif not isinstance(output_payload_dict["input_data"]["input_string"], list):
                errors.append("`input_string` field should be a list while got {}".format(
                    output_payload_dict["input_data"]["input_string"]
                ))
        if self._model.is_aoai_model():
            if "messages" not in output_payload_dict:
                errors.append(
                    "`messages` should be presented in the payload json.")
            elif not isinstance(output_payload_dict['messages'], list):
                errors.append(
                    "`messages` field in the payload should be a list."
                )
        return errors

    def _convert_python_pattern(self, origin_json_dict: Dict[str, Any]) -> Dict[str, Any]:
        placeholders = re.findall('###<[_a-zA-Z0-9 ]+>', self._batch_input_pattern)
        all_old_keys = set(["###<%s>" % k for k in origin_json_dict.keys()])
        new_json_string = self._batch_input_pattern
        for k in placeholders:
            assert k in all_old_keys, f"place holder {k} cannot be found in the input jsonl."
        for k, v in origin_json_dict.items():
            placeholder = "###<%s>" % k
            if placeholder in placeholders:
                if isinstance(v, str):
                    # replace special characters to avoid error when doing json deserialization
                    new_json_string = new_json_string.replace(placeholder, json.dumps(v)[1:-1])
                elif isinstance(v, dict) or isinstance(v, list):
                    new_json_string = new_json_string.replace(placeholder, json.dumps(v))
                else:
                    new_json_string = new_json_string.replace(placeholder, str(v))
        return json.loads(new_json_string)

    @staticmethod
    def from_args(args):
        """Init the class from args."""
        return EndpointDataPreparer(args.model_type, args.batch_input_pattern)
