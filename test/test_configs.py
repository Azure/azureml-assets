# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test Config classes."""

from pathlib import Path

import azureml.assets as assets

RESOURCES_DIR = Path("resources/config")


def test_comparisons():
    """Test AssetConfig comparisons."""
    this_dir = Path(__file__).parent
    resources = this_dir / RESOURCES_DIR

    # Load assets
    env1_1 = assets.AssetConfig(resources / "env1-1" / assets.DEFAULT_ASSET_FILENAME)
    env1_1_copy = assets.AssetConfig(resources / "env1-1" / assets.DEFAULT_ASSET_FILENAME)
    env1_2 = assets.AssetConfig(resources / "env1-2" / assets.DEFAULT_ASSET_FILENAME)
    env1_100 = assets.AssetConfig(resources / "env1-1.0.0" / assets.DEFAULT_ASSET_FILENAME)
    env1_101 = assets.AssetConfig(resources / "env1-1.0.1" / assets.DEFAULT_ASSET_FILENAME)
    env1_auto = assets.AssetConfig(resources / "env1-auto" / assets.DEFAULT_ASSET_FILENAME)

    assert env1_1 == env1_1
    assert env1_1 == env1_1_copy
    assert env1_1 < env1_2
    assert env1_100 < env1_101
    try:
        env1_1 < env1_auto
    except ValueError as e:
        assert "Cannot compare" in str(e)

    envs = [env1_1, env1_2, env1_100, env1_101]
    sorted_envs = sorted(envs)
    assert sorted_envs == [env1_1, env1_100, env1_101, env1_2]


def test_dependencies():
    """Test Spec dependency retrieval."""
    this_dir = Path(__file__).parent
    resources = this_dir / RESOURCES_DIR

    # Load assets
    comp_command = assets.AssetConfig(resources / "comp-command" / assets.DEFAULT_ASSET_FILENAME)
    comp_parallel = assets.AssetConfig(resources / "comp-parallel" / assets.DEFAULT_ASSET_FILENAME)
    comp_pipeline = assets.AssetConfig(resources / "comp-pipeline" / assets.DEFAULT_ASSET_FILENAME)

    assert comp_command.spec_as_object().dependencies == {
        assets.AssetType.ENVIRONMENT: {"azureml:environment:0.0.1"}
    }
    assert comp_parallel.spec_as_object().dependencies == {
        assets.AssetType.ENVIRONMENT: {"azureml:environment:0.0.1"}
    }
    assert comp_pipeline.spec_as_object().dependencies == {
        assets.AssetType.COMPONENT: {"azureml:component_1:0.0.1", "azureml:component_2:0.0.1"}
    }
