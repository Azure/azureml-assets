def pytest_generate_tests(metafunc):
    params = [
        ("build-bad", False),
        ("build-test-bad", False),
        ("build-test-good", True),
        ("pre-built-good", True),
        ("pre-built-latest-bad", True),
        ("pre-built-latest-good", True),
    ]
    if 'test_subdir' in metafunc.fixturenames:
        metafunc.parametrize('test_subdir', [p[0] for p in params])
    if 'expected' in metafunc.fixturenames:
        metafunc.parametrize('expected', [p[1] for p in params])
