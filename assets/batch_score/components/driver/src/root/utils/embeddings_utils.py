# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Embeddings utilities."""

import pandas as pd


def _convert_to_list_of_input_batches(
        data: pd.DataFrame,
        batch_size_per_request: int) -> "list[dict]":
    """Convert input data to a list of input batches."""
    """This method is specific for APIs that allow batching, currently only Embeddings.
    That means the data has the "input" column.

    Given a dataframe and batch size, convert the data into a list of dictionaries,
    where each element has an "input" list of strings equal* to the batch size.
    *The last element's "input" list of strings will have length in [1, batch_size_per_request].
    """
    numrows = len(data)
    list_of_input_batches = []

    for i in range(0, numrows, batch_size_per_request):
        list_of_strings = data["input"][i: i + batch_size_per_request].values.tolist()
        payload_obj = {"input": list_of_strings}
        list_of_input_batches.append(payload_obj)
    return list_of_input_batches
