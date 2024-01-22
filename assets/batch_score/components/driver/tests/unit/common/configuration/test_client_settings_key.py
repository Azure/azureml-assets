# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for client settings key."""

from src.batch_score.common.configuration.client_settings import ClientSettingsKey


def test_string_equality_as_dictionary_key():
    '''Ensure that the enum is equivalent to its string value when used as a dictionary key.'''
    # See question: https://stackoverflow.com/questions/58608361/string-based-enum-in-python
    # And answer: https://stackoverflow.com/a/58608362

    client_settings = {'CONCURRENCY_ADJUSTMENT_INTERVAL': 5}
    value = client_settings.get(ClientSettingsKey.CONCURRENCY_ADJUSTMENT_INTERVAL)
    assert value == 5
