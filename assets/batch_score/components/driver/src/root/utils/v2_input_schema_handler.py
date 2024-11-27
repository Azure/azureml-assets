# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the new (V2) schema input handler."""

import pandas as pd

from .input_handler import InputHandler


class V2InputSchemaHandler(InputHandler):
    """Defines a class to handle the new input schema."""

    def convert_input_to_requests(
            self,
            data: pd.DataFrame,
            additional_properties: str = None,
            batch_size_per_request: int = 1) -> "list[str]":
        """Convert the new schema input pandas DataFrame to a list of payload strings."""
        body_details = []
        for _, row in data.iterrows():
            body = row['body']
            del body['model']
            body['custom_id'] = row['custom_id']
            body_details.append(body)
        original_schema_df = pd.DataFrame(body_details)
        return self._convert_to_list(original_schema_df, additional_properties, batch_size_per_request)
