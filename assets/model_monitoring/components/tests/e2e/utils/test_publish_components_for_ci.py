# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utility that will upload components in special way for use in github CI."""

import pytest


class TestPublishComponentsForCI():
    def test_publish_components(self, upload_component_version_file):
        """Init calls fixture to publish components."""
        pass
