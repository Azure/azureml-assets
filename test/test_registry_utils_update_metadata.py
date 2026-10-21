# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test registry_utils.update_metadata system_metadata handling."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from azureml.assets.config import AssetType
from azureml.assets.deployment_config import AssetVersionUpdate
from azureml.assets.model import registry_utils
from azureml.assets.model.registry_utils import update_metadata


def _run(existing_system_metadata, system_metadata_update, raise_on_error=False, create_error=None):
    asset = SimpleNamespace(tags={}, properties={}, description=None, _system_metadata=existing_system_metadata)
    operations = MagicMock()
    operations.get.return_value = asset
    if create_error:
        operations.create_or_update.side_effect = create_error
    update = AssetVersionUpdate(versions=["1"], system_metadata=system_metadata_update)
    with patch.object(registry_utils.RegistryUtils, "get_operations_from_type", return_value=operations):
        update_metadata(name="model1", version="1", update=update, ml_client=MagicMock(),
                        asset_type=AssetType.MODEL, raise_on_error=raise_on_error)
    return asset, operations


def test_add_when_registry_returns_none_system_metadata():
    """Adding to an asset whose system_metadata is None sends only the added keys."""
    asset, operations = _run(None, {"add": {"modelCapabilities": ["agentsV2"]}})

    operations.create_or_update.assert_called_once_with(asset)
    assert asset._system_metadata == {"modelCapabilities": ["agentsV2"]}


def test_add_merges_with_existing_system_metadata():
    """Adding keeps existing keys and stores list values as lists."""
    asset, operations = _run({"summary": "s"}, {"add": {"modelCapabilities": ["agentsV2"]}})

    operations.create_or_update.assert_called_once()
    assert asset._system_metadata == {"summary": "s", "modelCapabilities": ["agentsV2"]}


def test_no_change_skips_update():
    """An add that matches existing values does not call the service."""
    _, operations = _run({"modelCapabilities": ["agentsV2"]}, {"add": {"modelCapabilities": ["agentsV2"]}})

    operations.create_or_update.assert_not_called()


def test_delete_with_none_system_metadata_skips_update():
    """Deleting from an asset with no returned system_metadata is a no-op."""
    _, operations = _run(None, {"delete": ["summary"]})

    operations.create_or_update.assert_not_called()


def test_failure_logged_by_default():
    """Update failures are only logged unless raise_on_error is set."""
    _run(None, {"add": {"modelCapabilities": ["agentsV2"]}}, create_error=RuntimeError("boom"))


def test_failure_raised_with_raise_on_error():
    """Update failures propagate when raise_on_error is set."""
    with pytest.raises(RuntimeError, match="boom"):
        _run(None, {"add": {"modelCapabilities": ["agentsV2"]}}, raise_on_error=True,
             create_error=RuntimeError("boom"))
