# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for scoring request."""

import json
import uuid
from .common.json_encoder_extensions import BatchComponentJSONEncoder
from ..request_modification.input_transformer import InputTransformer


class ScoringRequest:
    """Class for scoring request."""

    __BATCH_REQUEST_METADATA = "_batch_request_metadata"

    def __init__(
            self,
            original_payload: str,
            request_input_transformer: InputTransformer = None,
            logging_input_transformer: InputTransformer = None):
        """Init method."""
        self.__internal_id: str = str(uuid.uuid4())
        self.__original_payload = original_payload
        self.__original_payload_obj = json.loads(original_payload)

        # Used in checking if the max_retry_time_interval has been exceeded
        self.scoring_duration: int = 0

        # Duplicate original payload obj, to not change the original
        self.__cleaned_payload_obj = json.loads(original_payload)
        self.__loggable_payload_obj = json.loads(original_payload)

        # Scrub metadata from payload, if present.
        self.__request_metadata = self.__cleaned_payload_obj.pop(self.__BATCH_REQUEST_METADATA, None)

        # Apply any modifiers to the original payload
        if request_input_transformer:
            self.__cleaned_payload_obj = request_input_transformer.apply_modifications(self.__cleaned_payload_obj)

        if logging_input_transformer:
            self.__loggable_payload_obj = logging_input_transformer.apply_modifications(self.__loggable_payload_obj)

        self.__cleaned_payload = json.dumps(self.__cleaned_payload_obj, cls=BatchComponentJSONEncoder)
        self.__loggable_payload = json.dumps(self.__loggable_payload_obj, cls=BatchComponentJSONEncoder)

        # tuple(endpoint_base_url: str, response_status: int, model_response_code: str, is_retriable: bool)
        self.request_history: list[tuple[str, int, str, bool]] = []  # Stack
        self.estimated_cost: int = None
        self.total_wait_time: int = 0
        self.retry_count: int = 0

    # read-only
    @property
    def internal_id(self):
        """Internal id."""
        return self.__internal_id

    # read-only
    @property
    def original_payload(self):
        """Original payload."""
        return self.__original_payload

    # read-only
    @property
    def original_payload_obj(self):
        """Original payload object."""
        return self.__original_payload_obj

    # read-only
    @property
    def cleaned_payload_obj(self):
        """Cleaned payload object."""
        return self.__cleaned_payload_obj

    # read-only
    @property
    def cleaned_payload(self):
        """Cleaned payload."""
        return self.__cleaned_payload

    # read-only
    @property
    def loggable_payload(self):
        """Loggable payload."""
        return self.__loggable_payload

    # read-only
    @property
    def request_metadata(self):
        """Request metadata."""
        return self.__request_metadata
