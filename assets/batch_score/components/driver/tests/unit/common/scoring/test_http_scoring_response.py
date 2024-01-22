# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for HTTP scoring response."""

from src.batch_score.common.scoring.http_scoring_response import HttpScoringResponse


def test_get_model_response_code():
    # Arrange
    expected_model_response_code = 200
    headers = {'ms-azureml-model-error-statuscode': expected_model_response_code}

    http_response = HttpScoringResponse(headers=headers)

    # Act
    actual_model_response_code = http_response.get_model_response_code()

    # Assert
    assert expected_model_response_code == actual_model_response_code


def test_get_model_response_reason():
    # Arrange
    expected_model_response_reason = 'Failed model'
    headers = {'ms-azureml-model-error-reason': expected_model_response_reason}

    http_response = HttpScoringResponse(headers=headers)

    # Act
    actual_model_response_reason = http_response.get_model_response_reason()

    # Assert
    assert expected_model_response_reason == actual_model_response_reason
