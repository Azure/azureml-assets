from pathlib import Path

import azureml.assets as assets


def test_test_assets():
    this_dir = Path(__file__).parent

    assert assets.test_assets(
        this_dir / "resources/good-assets",
        assets.DEFAULT_ASSET_FILENAME,
        this_dir / "../scripts/test-requirements.txt",
        [])

    assert not assets.test_assets(
        this_dir / "resources/bad-assets",
        assets.DEFAULT_ASSET_FILENAME,
        this_dir / "../scripts/test-requirements.txt",
        [])
