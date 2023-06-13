# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test DeployConfig classes."""

from pathlib import Path

import azureml.assets as assets
from azureml.assets import AssetType

RESOURCES_DIR = Path("resources/deploy_config")


def test_load():
    """Test DeployConfig comparisons."""
    this_dir = Path(__file__).parent
    resources = this_dir / RESOURCES_DIR

    # Load deploy config
    config = assets.DeploymentConfig.load(resources / "good.yaml")

    # Test create
    assert config.create[AssetType.COMPONENT] == ["component1", "component2"]
    assert config.create[AssetType.MODEL] == ["model1", "model2"]

    # Test update
    env_updates = config.update[AssetType.ENVIRONMENT]
    print(env_updates)
    assert [u.name for u in env_updates] == ["environment1", "environment2"]
    for env_update in env_updates:
        update = env_update.updates[0]
        assert update.description == f"Test description for {env_update.name}."
        if env_update.name == "environment1":
            assert update.all_versions is True
            assert update.versions is None
            assert update.tags.add == {"AddMe": "Value"}
            assert update.tags.delete == ["DeleteMe"]
            assert update.stage == "Active"
        elif env_update.name == "environment2":
            assert update.all_versions is False
            assert update.versions == ["1", "2"]
            assert update.tags.replace == {"ReplaceMe": "Value"}
            assert update.stage == "Archived"

    # Test delete
    comp_deletes = config.delete[AssetType.COMPONENT]
    assert [d.name for d in comp_deletes] == ["component3", "component4"]
    for comp_delete in comp_deletes:
        delete = comp_delete.deletes[0]
        if env_update.name == "component3":
            assert delete.all_versions is True
            assert delete.delete_container is False
        elif env_update.name == "component4":
            assert delete.versions == ["3", "4"]
            assert delete.delete_container is True
