# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test environment folder scripts."""

from azureml.assets.environment import pin_packages
from azureml.assets.environment.pin_package_versions import (
    create_package_finder,
    get_latest_package_version,
    PYPI_URL,
)

package_finder = create_package_finder([PYPI_URL])


def test_pin_packages_no_extras():
    contents = "azureml-core=={{latest-pypi-version}}"
    expected = f"azureml-core=={get_latest_package_version('azureml-core', package_finder)}"
    assert pin_packages(contents) == expected


def test_pin_packages_with_extras():
    contents = "azureml-metrics[all]=={{latest-pypi-version}}"
    expected = f"azureml-metrics[all]=={get_latest_package_version('azureml-metrics', package_finder)}"
    assert pin_packages(contents) == expected
