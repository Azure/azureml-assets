# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test release folder scripts."""

import azureml.assets.util as util
import pytest
from pathlib import Path
from typing import List, Tuple

import azureml.assets as assets

RESOURCES_DIR = Path("resources/release")


@pytest.mark.parametrize(
    "test_subdir,expected",
    [
        ("environment-in-parent-dir", [
            "asset.yaml",
            "spec.yaml",
            "environment.yaml",
            "../src/context/conda_dependencies.yaml",
            "../src/context/Dockerfile",
        ]),
        ("environment-in-subdir", [
            "asset.yaml",
            "spec.yaml",
            "environment.yaml",
            "context/conda_dependencies.yaml",
            "context/Dockerfile"
        ]),
        ("component-in-parent-dir", [
            "asset.yaml",
            "spec.yaml",
            "../src/code/run.py"
        ])
    ]
)
def test_release_paths(test_subdir: str, expected: List[Path]):
    """Test release paths."""
    this_dir = Path(__file__).parent
    test_dir = this_dir / RESOURCES_DIR / test_subdir

    asset_config = assets.AssetConfig(test_dir / assets.DEFAULT_ASSET_FILENAME)
    expected_paths = [test_dir / p for p in expected]

    assert sorted(asset_config.release_paths) == sorted(expected_paths)


@pytest.mark.parametrize(
    "test_subdir,expected",
    [
        ("environment-in-parent-dir",
            ("..", [
                "asset.yaml",
                "spec.yaml",
                "environment.yaml",
                "../src/context/conda_dependencies.yaml",
                "../src/context/Dockerfile"
            ])),
        ("environment-in-subdir",
            (".", [
                "asset.yaml",
                "spec.yaml",
                "environment.yaml",
                "context/conda_dependencies.yaml",
                "context/Dockerfile"
            ])),
        ("component-in-parent-dir",
            ("..", [
                "asset.yaml",
                "spec.yaml",
                "../src/code/run.py"
            ]))
    ]
)
def test_find_common_directory(test_subdir: str, expected: Tuple[Path, List[Path]]):
    this_dir = Path(__file__).parent
    test_dir = this_dir / RESOURCES_DIR / test_subdir

    # Convert release paths to common dir and relative paths
    asset_config = assets.AssetConfig(test_dir / assets.DEFAULT_ASSET_FILENAME)
    release_paths = asset_config.release_paths
    common_dir, release_paths_relative_to_common_dir = util.find_common_directory(release_paths)

    # Reformat expected results
    expected_common_dir, expected_paths = expected
    expected_common_dir = Path(test_dir / expected_common_dir).resolve()
    expected_paths = [(test_dir / p).resolve().relative_to(expected_common_dir) for p in expected_paths]

    assert common_dir.samefile(expected_common_dir)
    assert sorted(release_paths_relative_to_common_dir) == sorted(expected_paths)
