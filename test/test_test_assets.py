from pathlib import Path

import azureml.assets as assets


# TODO: Make this work

def test_test_assets():
    this_dir = Path(__file__).parent
    assert assets.test_assets(
        this_dir / "resources/test-assets",
        assets.DEFAULT_ASSET_FILENAME,
        this_dir / "../scripts/test-requirements.txt",
        [])
