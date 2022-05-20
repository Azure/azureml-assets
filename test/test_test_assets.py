from pathlib import Path
import pytest

import azureml.assets as assets

RESOURCES_DIR = Path("resources")


@pytest.mark.parametrize(
    "test_subdir,expected",
    [("good-assets-with-requirements", True), ("bad-assets", False), ("mixed-assets", False)]
)
def test_test_assets(test_subdir: str, expected: bool):
    this_dir = Path(__file__).parent

    assert assets.test_assets(
        this_dir / RESOURCES_DIR / test_subdir,
        assets.DEFAULT_ASSET_FILENAME,
        this_dir / "../scripts/test-requirements.txt",
        []) == expected
