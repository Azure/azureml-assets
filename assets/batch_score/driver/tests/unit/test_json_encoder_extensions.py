# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for JSON encoder extensions."""

import json

import src.batch_score.utils.json_encoder_extensions as json_encoder_extensions


def test_unconfigured_get_default_encoder(mock_get_logger):
    encoder = json_encoder_extensions.get_default_encoder()
    assert encoder.ensure_ascii is True

    serialized_obj = json.dumps("こんにちは世界!", cls=json_encoder_extensions.BatchComponentJSONEncoder)
    assert serialized_obj == '"\\u3053\\u3093\\u306b\\u3061\\u306f\\u4e16\\u754c!"'


def test_configured_get_default_encoder(mock_get_logger):
    json_encoder_extensions.setup_encoder(ensure_ascii=False)

    encoder = json_encoder_extensions.get_default_encoder()
    assert encoder.ensure_ascii is False

    serialized_obj = json.dumps("こんにちは世界!", cls=json_encoder_extensions.BatchComponentJSONEncoder)
    assert serialized_obj == '"こんにちは世界!"'


def test_configuration_instantiation_is_singular(mock_get_logger):
    json_encoder_extensions.setup_encoder(ensure_ascii=False)
    first_encoder = json_encoder_extensions.get_default_encoder()

    # Second configuration should be ignored, since first configuration already exists
    json_encoder_extensions.setup_encoder(ensure_ascii=True)
    second_encoder = json_encoder_extensions.get_default_encoder()

    assert first_encoder.ensure_ascii == second_encoder.ensure_ascii


def test_derived_numpy_array_encoder(mock_get_logger):
    import numpy

    ensure_ascii_scenario = False
    json_encoder_extensions.setup_encoder(ensure_ascii=ensure_ascii_scenario)

    numpy_encoder = json_encoder_extensions.NumpyArrayEncoder()
    assert numpy_encoder.ensure_ascii == ensure_ascii_scenario

    serialized_obj = json.dumps("こんにちは世界!", cls=json_encoder_extensions.NumpyArrayEncoder)
    assert serialized_obj == '"こんにちは世界!"'

    arr = numpy.array([0, 1, 2])
    serialized_obj = json.dumps(arr, cls=json_encoder_extensions.NumpyArrayEncoder)
    assert serialized_obj == "[0, 1, 2]"
