from pathlib import Path
import pytest

import azureml.assets as assets

RESOURCES_DIR = Path("resources/validate")


@pytest.mark.parametrize(
    "test_subdir,expected",
    [("name-mismatch", False), ("version-mismatch", False), ("good-validation", True)]
)
def test_validate_assets(test_subdir: str, expected: bool):
    this_dir = Path(__file__).parent

    assert assets.validate_assets(
        this_dir / RESOURCES_DIR / test_subdir,
        assets.DEFAULT_ASSET_FILENAME) == expected
