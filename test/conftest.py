def pytest_addoption(parser):
    parser.addoption("--resource-group", action="store")
    parser.addoption("--registry", action="store")


def pytest_generate_tests(metafunc):
    resource_group_value = metafunc.config.option.resource_group
    if 'resource_group' in metafunc.fixturenames and resource_group_value is not None:
        metafunc.parametrize('resource_group', [resource_group_value])

    registry_value = metafunc.config.option.registry
    if 'registry' in metafunc.fixturenames and registry_value is not None:
        metafunc.parametrize('registry', [registry_value])
