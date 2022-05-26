def pytest_generate_tests(metafunc):
    params = [
        ("build-bad", False),
        ("build-test-bad", False),
        ("build-test-good", True),
        ("pre-built-good", True),
        ("pre-built-latest-bad", False),
        ("pre-built-latest-good", True),
    ]
    if 'subdir_expected_pair' in metafunc.fixturenames:
        metafunc.parametrize('subdir_expected_pair', params)
