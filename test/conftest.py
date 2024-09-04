# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test various pytest cases with ."""

def pytest_addoption(parser):
    """Add pytest options."""
    parser.addoption("--resource-group", action="store")
    parser.addoption("--registry", action="store")


def pytest_generate_tests(metafunc):
    """Generate test cases based on options."""
    resource_group_value = metafunc.config.option.resource_group
    if 'resource_group' in metafunc.fixturenames and resource_group_value is not None:
        metafunc.parametrize('resource_group', [resource_group_value])

    registry_value = metafunc.config.option.registry
    if 'registry' in metafunc.fixturenames and registry_value is not None:
        metafunc.parametrize('registry', [registry_value])

    if 'build_subdir_expected_pair' in metafunc.fixturenames:
        metafunc.parametrize('build_subdir_expected_pair', [
            ("build-bad", False),
            ("build-latest-regex-bad", False),
            ("build-latest-regex-good", True),
            ("build-test-bad", False),
            ("build-test-good", True),
            ("pre-built-good", True),
            ("pre-built-latest-bad", False),
            ("pre-built-latest-good", True),
        ])
