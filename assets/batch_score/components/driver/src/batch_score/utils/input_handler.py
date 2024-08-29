# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the input handler."""

import collections.abc
import json
import numpy
import pandas as pd

from abc import ABC, abstractmethod

from . import embeddings_utils as embeddings
from .json_encoder_extensions import NumpyArrayEncoder


class InputHandler(ABC):
    """An abstract class for handling input."""

    @abstractmethod
    def convert_input_to_requests(
            data: pd.DataFrame,
            additional_properties: str,
            batch_size_per_request: int) -> "list[str]":
        """Abstract input row formatting method."""
        pass

    def _add_properties_to_payload_object(
            self,
            payload_obj: dict,
            additional_properties_list: dict):
        if additional_properties_list is not None:
            for key, value in additional_properties_list.items():
                payload_obj[key] = value

    def _stringify_payload(self, payload_obj: dict) -> str:
        return json.dumps(payload_obj, cls=NumpyArrayEncoder)

    def _convert_to_list(
            self,
            data: pd.DataFrame,
            additional_properties: str = None,
            batch_size_per_request: int = 1) -> "list[str]":
        """Original function to convert the input panda data frame to a list of payload strings."""
        columns = data.keys()
        payloads = []
        additional_properties_list = None

        # Per https://platform.openai.com/docs/api-reference/
        int_forceable_properties = [
            "max_tokens",
            "n",
            "best_of",
            "batch_size",
            "classification_n_classes"]

        if additional_properties is not None:
            additional_properties_list = json.loads(additional_properties)

        if batch_size_per_request > 1:
            batched_payloads = embeddings._convert_to_list_of_input_batches(data, batch_size_per_request)
            payloads.extend(batched_payloads)
        else:
            for row in data.values.tolist():
                payload_obj: dict[str, any] = {}
                for indx, col in enumerate(columns):
                    payload_val = row[indx]
                    if isinstance(payload_val, collections.abc.Sequence) or isinstance(payload_val, numpy.ndarray) or \
                            not pd.isnull(payload_val):
                        if col in int_forceable_properties:
                            payload_val = int(payload_val)
                        payload_obj[col] = payload_val
                payloads.append(payload_obj)

        for idx, payload in enumerate(payloads):
            # Payloads are mutable. Update them with additional properties and put them back in the list as strings.
            self._add_properties_to_payload_object(payload, additional_properties_list)
            payloads[idx] = self._stringify_payload(payload)

        return payloads
