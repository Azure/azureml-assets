# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the original (V1) schema input handler."""

import pandas as pd

from .input_handler import InputHandler


class V1InputSchemaHandler(InputHandler):
    """Defines a class to handle the original input schema. This is used as the default input handler."""

    def convert_input_to_requests(
            self,
            data: pd.DataFrame,
            additional_properties: str = None,
            batch_size_per_request: int = 1) -> "list[str]":
        """Convert the original schema input pandas DataFrame to a list of payload strings."""
        return self._convert_to_list(data, additional_properties, batch_size_per_request)
