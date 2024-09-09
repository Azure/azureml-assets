# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utility that will upload components in special way for use in github CI."""

import pytest
import os


@pytest.mark.skipif(
    condition=os.path.exists('component_version/.version_upload'),
    reason="For local e2e pytest runs and in CI if we already uploaded the shared component version")
class TestPublishComponentsForCI():
    """Class for model-monitoring-ci workflow to upload components and data once before splitting tests to runners."""

    def test_publish_components(self, upload_component_version_file, cleanup_previous_e2e_tests):
        """Init calls fixture to publish components."""
        assert os.path.exists('component_version/.version_upload')
