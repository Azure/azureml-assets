# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

import pytest
from generation_safety_quality.annotation_compute_histogram.run import (
    _check_and_format_azure_endpoint_url,
    AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN,
    AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE)
from shared_utilities.momo_exceptions import InvalidInputError


@pytest.mark.unit
class TestGSQHistogram:
    """Test class for GSQ histogram component and utilities."""

    def test_gsq_invalid_deployment_url(self):
        """Test _check_and_format_azure_endpoint_url method in GSQ component."""
        url_pattern = AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN
        domain_pattern_re = AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE
        version = "2022-12-01"
        model = "test_model"
        invalid_url = "https://invalidurl.com"
        with pytest.raises(InvalidInputError):
            _check_and_format_azure_endpoint_url(
                url_pattern, domain_pattern_re, invalid_url,
                version, model)
        # this was the url causing the error
        cog_url = "australiaeast.api.cognitive.microsoft.com"
        with pytest.raises(InvalidInputError):
            _check_and_format_azure_endpoint_url(
                url_pattern, domain_pattern_re, cog_url, version, model)

    def test_gsq_valid_deployment_url(self):
        """Test _check_and_format_azure_endpoint_url method in GSQ component."""
        url_pattern = AZURE_OPENAI_API_DEPLOYMENT_URL_PATTERN
        domain_pattern_re = AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE
        version = "2022-12-01"
        model = "test_model"
        valid_url = "abc.openai.azure.com"
        formatted_url = _check_and_format_azure_endpoint_url(
            url_pattern, domain_pattern_re, valid_url, version, model)
        expected_format = f"https://{valid_url}/openai/deployments/{model}?api-version={version}"
        assert formatted_url == expected_format
