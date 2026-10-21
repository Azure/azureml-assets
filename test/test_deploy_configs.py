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
    assert config.should_create(AssetType.COMPONENT, "component1") is True

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
            assert update.update_on_create.enabled is True
            assert update.update_on_create.clouds == ["public"]
            assert update.tags.add == {"AddMe": "Value"}
            assert update.tags.delete == ["DeleteMe"]
            assert update.stage == "Active"
        elif env_update.name == "environment2":
            assert update.all_versions is False
            assert update.versions == ["1", "2"]
            assert update.update_on_create is None
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


def test_load_system_metadata_typed_values():
    """Test that system_metadata values keep their YAML types as plain JSON types."""
    resources = Path(__file__).parent / RESOURCES_DIR

    config = assets.DeploymentConfig.load(resources / "system_metadata.yaml")

    model_updates = {u.name: u.updates[0] for u in config.update[AssetType.MODEL]}
    add = model_updates["model1"].system_metadata.add
    assert add == {
        "modelCapabilities": ["agentsV2"],
        "deprecation": {"deprecationDate": "2026-11-02"},
        "summary": "Plain string value",
    }
    assert type(add["modelCapabilities"]) is list
    assert type(add["deprecation"]) is dict
    assert model_updates["model2"].system_metadata.delete == ["summary"]
