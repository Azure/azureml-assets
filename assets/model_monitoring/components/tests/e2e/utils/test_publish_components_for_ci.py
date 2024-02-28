# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utility that will upload components in special way for use in github CI."""


class TestPublishComponentsForCI():
    """Class for model-monitoring-ci workflow to upload components and data once before splitting tests to runners."""

    def test_publish_components(self, upload_component_version_file):
        """Init calls fixture to publish components."""
        pass
