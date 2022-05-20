from pathlib import Path

import azureml.assets as assets


# TODO: Make this work

def test_success():
    this_dir = Path(__file__).parent
    assets.test_assets(
        this_dir / "test-assets",
        assets.DEFAULT_ASSET_FILENAME,
        this_dir / "../scripts/test-requirements.txt",
        [])
    assert False


def test_failure():
    assert False
