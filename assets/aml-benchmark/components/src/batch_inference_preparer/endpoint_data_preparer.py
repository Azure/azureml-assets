# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from typing import Any, Dict
import json
import re


class EndpointDataPreparer:
    """Endpoint data preparer class"""
    def __init__(self, model_type: str, batch_input_pattern: str):
        self._model_type = model_type
        self._batch_input_pattern = batch_input_pattern

    def convert_input_dict(self, origin_json_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert input dict to the corresponding payload"""
        return self._convert_python_pattern(origin_json_dict)
    
    def validate_output(self, output_payload_dict: Dict[str, Any]):
        """Validate the output payload."""
        errors = []
        if self._model_type.lower == "llama":
            if "parameters" not in output_payload_dict["input_data"]:
                errors.append("`parameters` should be presented in the payload json.")
        return errors
    
    def _convert_python_pattern(self, origin_json_dict: Dict[str, Any]) -> Dict[str, Any]:
        placeholders = re.findall('###<[\w ]+>', self._batch_input_pattern)
        all_old_keys = set(["###<%s>" % k for k in origin_json_dict.keys()])
        new_json_string = self._batch_input_pattern
        for k in placeholders:
            assert k in all_old_keys, f"place holder {k} cannot be found in the input jsonl."
        for k, v in origin_json_dict.items():
            placeholder = "###<%s>" % k
            if placeholder in placeholders:
                if isinstance(v, str):
                    # replace special characters to avoid error when doing json deserialization
                    new_json_string = new_json_string.replace(placeholder, v.replace('\\', '\\\\').replace("\n", "\\n").replace('"', '\\"'))
                elif isinstance(v, dict) or isinstance(v, list):
                    new_json_string = new_json_string.replace(placeholder, json.dumps(v))
                else:
                    new_json_string = new_json_string.replace(placeholder, str(v))
        print(new_json_string)
        return json.loads(new_json_string)

    @staticmethod
    def from_args(args):
        """Init the class from args."""
        return EndpointDataPreparer(args.model_type, args.batch_input_pattern)
