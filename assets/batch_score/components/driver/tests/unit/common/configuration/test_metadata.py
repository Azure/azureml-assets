# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for metadata class."""

import pytest

from src.batch_score.common.configuration.metadata import Metadata

from tests.fixtures.configuration import TEST_COMPONENT_NAME, TEST_COMPONENT_VERSION


def test_get_metadata_success_component_name_without_file_extension():
    # Act
    metadata_payload = {
        "component_name": TEST_COMPONENT_NAME,
        "component_version": TEST_COMPONENT_VERSION,
    }
    result = Metadata.get_metadata(metadata_payload)

    # Assert
    assert result.component_name == TEST_COMPONENT_NAME
    assert result.component_version == TEST_COMPONENT_VERSION


@pytest.mark.parametrize(
    'component_name',
    [
        f"{TEST_COMPONENT_NAME}.yml",
        f"{TEST_COMPONENT_NAME}.yaml",
    ]
)
def test_get_metadata_success_component_name_with_file_extension(component_name):
    # Act
    metadata_payload = {
        "component_name": component_name,
        "component_version": TEST_COMPONENT_VERSION,
    }
    result = Metadata.get_metadata(metadata_payload)

    # Assert
    assert result.component_name == TEST_COMPONENT_NAME
    assert result.component_version == TEST_COMPONENT_VERSION
