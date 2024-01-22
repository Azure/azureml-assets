import json
import uuid

from ...utils.json_encoder_extensions import BatchComponentJSONEncoder
from ..post_processing.mini_batch_context import MiniBatchContext
from ..request_modification.input_transformer import InputTransformer
from .scoring_attempt import ScoringAttempt


class ScoringRequest:
    __BATCH_REQUEST_METADATA = "_batch_request_metadata"
    __REQUEST_METADATA = "request_metadata"

    def __init__(self,
                 original_payload: str,
                 input_to_request_transformer: InputTransformer = None,
                 input_to_log_transformer: InputTransformer = None,
                 mini_batch_context: MiniBatchContext = None):
        self.__internal_id: str = str(uuid.uuid4())
        self.__original_payload = original_payload
        self.__original_payload_obj = json.loads(original_payload)
        self.__input_to_request_transformer = input_to_request_transformer
        self.__input_to_log_transformer = input_to_log_transformer

        # Used in checking if the max_retry_time_interval has been exceeded
        self.scoring_duration: int = 0

        # Duplicate original payload obj, to not change the original
        self.__cleaned_payload_obj = json.loads(original_payload)
        self.__loggable_payload_obj = json.loads(original_payload)

        # Apply any modifiers to the original payload
        if self.__input_to_request_transformer:
            self.__cleaned_payload_obj = self.__input_to_request_transformer.apply_modifications(self.__cleaned_payload_obj)

        if self.__input_to_log_transformer:
            self.__loggable_payload_obj = self.__input_to_log_transformer.apply_modifications(self.__loggable_payload_obj)

        # Pop _batch_request_metadata property from payload, if present 
        # Override with request_metadata, if present
        # In case both exist, we call pop twice to make sure these properties
        # are removed from the cleaned payload object
        # These properties do not need to be sent to the model & will be added to the output file directly
        self.__request_metadata = self.__cleaned_payload_obj.pop(self.__BATCH_REQUEST_METADATA, None)
        self.__request_metadata = self.__cleaned_payload_obj.pop(self.__REQUEST_METADATA, self.__request_metadata)

        self.__cleaned_payload = json.dumps(self.__cleaned_payload_obj, cls=BatchComponentJSONEncoder)
        self.__loggable_payload = json.dumps(self.__loggable_payload_obj, cls=BatchComponentJSONEncoder)

        self.__estimated_cost: int = None
        self.__estimated_tokens_per_item_in_batch: tuple[int] = ()
        self.__segment_id: int = None
        self.__scoring_url: str = None

        self.mini_batch_context: MiniBatchContext = mini_batch_context
        self.request_history: list[ScoringAttempt] = [] # Stack
        self.retry_count: int = 0
        self.total_wait_time: int = 0

    # read-only
    @property
    def internal_id(self):
        return self.__internal_id

    # read-only
    @property
    def original_payload(self):
        return self.__original_payload

    # read-only
    @property
    def original_payload_obj(self):
        return self.__original_payload_obj
    
    # read-only
    @property
    def cleaned_payload_obj(self):
        return self.__cleaned_payload_obj

    # read-only
    @property
    def cleaned_payload(self):
        return self.__cleaned_payload

    # read-only
    @property
    def loggable_payload(self):
        return self.__loggable_payload

    # read-only
    @property
    def request_metadata(self):
        return self.__request_metadata

    # read-only
    @property
    def estimated_cost(self) -> int:
        return self.__estimated_cost or sum(self.__estimated_tokens_per_item_in_batch)
    
    # read-only
    @property
    def scoring_url(self) -> str:
        return self.__scoring_url
    
     # read-only
    @property
    def segment_id(self) -> int:
        return self.__segment_id

    @estimated_cost.setter
    def estimated_cost(self, cost: int):
        self.__estimated_cost = cost

    @scoring_url.setter
    def scoring_url(self, url: str):
        self.__scoring_url = url

    @segment_id.setter
    def segment_id(self, id: int):
        self.__segment_id = id

    # read-only
    @property
    def estimated_token_counts(self) -> "tuple[int]":
        return self.__estimated_tokens_per_item_in_batch

    @estimated_token_counts.setter
    def estimated_token_counts(self, token_counts: "tuple[int]"):
        self.__estimated_tokens_per_item_in_batch = token_counts

    def copy_with_new_payload(self, payload):
        '''Creates a copy of the ScoringRequest using the existing transformers on a new payload.'''
        return ScoringRequest(
            original_payload=payload,
            input_to_request_transformer=self.__input_to_request_transformer,
            input_to_log_transformer=self.__input_to_log_transformer,
            mini_batch_context=self.mini_batch_context,
        )
