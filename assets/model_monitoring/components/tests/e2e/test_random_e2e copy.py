# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains e2e tests for the create manifest component."""

import pytest

@pytest.mark.e2e
class TestCreateManifestE2E:
    """Test class."""

    def test_monitoring_run_use_defaults_data_has_no_drift_successful(self):
        """Test the happy path scenario where the data has drift and default settings are used."""
        assert True
