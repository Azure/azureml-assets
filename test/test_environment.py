# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test environment folder scripts."""

import requests
from azureml.assets.environment import pin_packages
from azureml.assets.environment.pin_package_versions import (
    create_package_finder,
    get_latest_package_version,
    PYPI_URL,
)

package_finder = create_package_finder([PYPI_URL])


def test_pin_packages_no_extras():
    """Test pin packages without extras."""
    contents = "azureml-core=={{latest-pypi-version}}"
    expected = f"azureml-core=={get_latest_package_version('azureml-core', package_finder)}"
    assert pin_packages(contents) == expected


def test_pin_packages_with_extras():
    """Test pin packages with extras."""
    contents = "azureml-metrics[all]=={{latest-pypi-version}}"
    expected = f"azureml-metrics[all]=={get_latest_package_version('azureml-metrics', package_finder)}"
    assert pin_packages(contents) == expected

def get_latest_version_from_pypi(package_name):
    """ Get latest package version from pypi"""
    url = f"https://pypi.org/pypi/{package_name}/json"
    latest_version = requests.get(url).json()["info"]["version"]
    return latest_version


def test_yanked_version_exclusion():
    """ Test latest-pypi-version to exclude yanked versions"""
    package_names = ["azureml-core", "azureml-acft-image-components", "azureml-metrics"]
    for package_name in package_names:
        expected = get_latest_version_from_pypi(package_name)
        actual = get_latest_package_version(package_name, package_finder)
        assert expected == actual
